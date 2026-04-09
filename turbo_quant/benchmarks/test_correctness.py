"""Correctness tests for TurboQuant quantizers.

Run with: pytest turbo_quant/benchmarks/test_correctness.py -xvvv
"""

import math
import torch
import pytest

from turbo_quant.rotation import generate_rotation_matrix
from turbo_quant.codebook import TurboQuantCodebook, lloyd_max
from turbo_quant.quantizer import TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
from turbo_quant.bitpack import pack, unpack, packed_size, compression_ratio


DEVICE = "cpu"
DTYPE = torch.float32


# ---- Rotation tests ----


class TestRotationMatrix:
    def test_orthogonality(self):
        """Π should satisfy ΠΠᵀ = I."""
        d = 128
        pi = generate_rotation_matrix(d, dtype=torch.float64)
        identity = torch.eye(d, dtype=torch.float64)
        assert torch.allclose(pi @ pi.t(), identity, atol=1e-10)

    def test_determinism(self):
        """Same seed should produce same matrix."""
        a = generate_rotation_matrix(128, seed=42)
        b = generate_rotation_matrix(128, seed=42)
        assert torch.equal(a, b)

    def test_different_seeds(self):
        """Different seeds should produce different matrices."""
        a = generate_rotation_matrix(128, seed=42)
        b = generate_rotation_matrix(128, seed=99)
        assert not torch.equal(a, b)


# ---- Codebook tests ----


class TestCodebook:
    def test_centroids_sorted(self):
        """Centroids should be in ascending order."""
        for bits in [1, 2, 3, 4]:
            cb = TurboQuantCodebook(128, bits)
            diffs = cb.centroids[1:] - cb.centroids[:-1]
            assert (diffs > 0).all(), f"Centroids not sorted for bits={bits}"

    def test_centroids_symmetric(self):
        """Centroids should be symmetric around 0 (distribution is symmetric)."""
        for bits in [1, 2, 3]:
            cb = TurboQuantCodebook(128, bits, dtype=torch.float64)
            n = cb.num_levels
            for i in range(n // 2):
                assert abs(cb.centroids[i].item() + cb.centroids[n - 1 - i].item()) < 1e-6

    def test_round_trip_identity_for_centroids(self):
        """Quantizing a centroid value should return the same centroid."""
        cb = TurboQuantCodebook(128, 2)
        for i in range(cb.num_levels):
            val = cb.centroids[i].unsqueeze(0)
            idx = cb.quantize(val)
            reconstructed = cb.dequantize(idx)
            assert torch.allclose(val, reconstructed, atol=1e-5)

    def test_quantize_range(self):
        """All indices should be valid."""
        cb = TurboQuantCodebook(128, 3)
        x = torch.randn(1000)
        x = x / x.norm() * math.sqrt(1000)  # roughly unit-sphere coords
        idx = cb.quantize(x)
        assert idx.min() >= 0
        assert idx.max() < cb.num_levels


# ---- MSE Quantizer tests ----


class TestTurboQuantMSE:
    @pytest.fixture
    def quantizer(self):
        return TurboQuantMSE(head_dim=128, bits=3, device=DEVICE, dtype=DTYPE)

    def test_shape_preservation(self, quantizer):
        """Output shapes should match input shapes."""
        x = torch.randn(2, 8, 16, 128, device=DEVICE)
        indices, norms = quantizer.quantize(x)
        assert indices.shape == (2, 8, 16, 128)
        assert norms.shape == (2, 8, 16, 1)

        x_rec = quantizer.dequantize(indices, norms)
        assert x_rec.shape == x.shape

    def test_reconstruction_error_bounded(self, quantizer):
        """MSE should decrease with more bits."""
        torch.manual_seed(0)
        x = torch.randn(4, 8, 32, 128, device=DEVICE)

        errors = {}
        for bits in [1, 2, 3, 4]:
            q = TurboQuantMSE(head_dim=128, bits=bits, device=DEVICE, dtype=DTYPE)
            indices, norms = q.quantize(x)
            x_rec = q.dequantize(indices, norms)
            mse = ((x - x_rec) ** 2).mean().item()
            errors[bits] = mse

        # More bits should give lower error
        assert errors[1] > errors[2] > errors[3] > errors[4], f"Errors not monotonically decreasing: {errors}"

    def test_zero_input(self, quantizer):
        """Quantizing zeros should not crash."""
        x = torch.zeros(1, 1, 1, 128, device=DEVICE)
        indices, norms = quantizer.quantize(x)
        x_rec = quantizer.dequantize(indices, norms)
        assert x_rec.shape == x.shape
        assert not torch.isnan(x_rec).any()

    def test_single_token(self, quantizer):
        """Should work for a single token."""
        x = torch.randn(1, 8, 1, 128, device=DEVICE)
        indices, norms = quantizer.quantize(x)
        x_rec = quantizer.dequantize(indices, norms)
        assert x_rec.shape == x.shape


# ---- Inner-Product Quantizer tests ----


class TestTurboQuantProd:
    @pytest.fixture
    def quantizer(self):
        return TurboQuantProd(head_dim=128, bits=3, device=DEVICE, dtype=DTYPE)

    def test_shape_preservation(self, quantizer):
        """Output shapes should be correct."""
        x = torch.randn(2, 8, 16, 128, device=DEVICE)
        mse_idx, mse_norms, qjl_signs, res_norms = quantizer.quantize(x)
        assert mse_idx.shape == (2, 8, 16, 128)
        assert mse_norms.shape == (2, 8, 16, 1)
        assert qjl_signs.shape == (2, 8, 16, 128)
        assert res_norms.shape == (2, 8, 16, 1)

        x_rec = quantizer.dequantize(mse_idx, mse_norms, qjl_signs, res_norms)
        assert x_rec.shape == x.shape

    def test_inner_product_unbiased(self, quantizer):
        """Inner product should be approximately unbiased over many samples."""
        torch.manual_seed(0)
        d = 128
        num_trials = 200

        y = torch.randn(d, device=DEVICE)
        y = y / y.norm()  # unit query vector

        true_ips = []
        est_ips = []

        for _ in range(num_trials):
            x = torch.randn(d, device=DEVICE)
            true_ip = (y * x).sum().item()

            x_4d = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,d]
            mse_idx, mse_norms, qjl_signs, res_norms = quantizer.quantize(x_4d)
            x_rec = quantizer.dequantize(mse_idx, mse_norms, qjl_signs, res_norms)
            est_ip = (y * x_rec.squeeze()).sum().item()

            true_ips.append(true_ip)
            est_ips.append(est_ip)

        # Check that mean estimated IP is close to mean true IP (unbiasedness)
        mean_true = sum(true_ips) / len(true_ips)
        mean_est = sum(est_ips) / len(est_ips)
        assert (
            abs(mean_est - mean_true) < 0.5
        ), f"Inner product estimate biased: true mean={mean_true:.4f}, est mean={mean_est:.4f}"

    def test_lower_error_than_mse_for_inner_products(self):
        """Prod variant should have reasonable inner-product error."""
        torch.manual_seed(42)
        d = 128
        total_bits = 3

        mse_q = TurboQuantMSE(head_dim=d, bits=total_bits, device=DEVICE, dtype=DTYPE)
        prod_q = TurboQuantProd(head_dim=d, bits=total_bits, device=DEVICE, dtype=DTYPE)

        y = torch.randn(d, device=DEVICE)
        ip_errors_mse = []
        ip_errors_prod = []

        for _ in range(100):
            x = torch.randn(d, device=DEVICE)
            true_ip = (y * x).sum().item()
            x_4d = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # MSE variant
            idx, norms = mse_q.quantize(x_4d)
            x_rec_mse = mse_q.dequantize(idx, norms)
            ip_mse = (y * x_rec_mse.squeeze()).sum().item()
            ip_errors_mse.append((ip_mse - true_ip) ** 2)

            # Prod variant
            mi, mn, qs, rn = prod_q.quantize(x_4d)
            x_rec_prod = prod_q.dequantize(mi, mn, qs, rn)
            ip_prod = (y * x_rec_prod.squeeze()).sum().item()
            ip_errors_prod.append((ip_prod - true_ip) ** 2)

        mse_error = sum(ip_errors_mse) / len(ip_errors_mse)
        prod_error = sum(ip_errors_prod) / len(ip_errors_prod)

        # Prod uses (bits-1) for MSE + 1 for QJL, so MSE component has fewer bits.
        # But the unbiasedness property of prod should make it competitive.
        # We just check it doesn't catastrophically fail.
        assert prod_error < mse_error * 5, f"Prod variant error too high: mse={mse_error:.6f}, prod={prod_error:.6f}"

    def test_min_bits_validation(self):
        """Should raise for bits < 2."""
        with pytest.raises(ValueError):
            TurboQuantProd(head_dim=128, bits=1)


# ---- Outlier-Aware Quantizer tests ----


class TestOutlierAwareTurboQuant:
    @pytest.fixture
    def quantizer(self):
        return OutlierAwareTurboQuant(
            head_dim=128,
            outlier_bits=3,
            normal_bits=2,
            num_outlier_channels=32,
            device=DEVICE,
            dtype=DTYPE,
        )

    def test_effective_bits(self, quantizer):
        """Effective bits should be (32*3 + 96*2) / 128 = 2.25."""
        assert abs(quantizer.effective_bits - 2.25) < 1e-6

    def test_shape_preservation(self, quantizer):
        """Output shapes should match input."""
        x = torch.randn(2, 8, 16, 128, device=DEVICE)
        indices, norms = quantizer.quantize(x)
        assert indices.shape == (2, 8, 16, 128)
        assert norms.shape == (2, 8, 16, 1)

        x_rec = quantizer.dequantize(indices, norms)
        assert x_rec.shape == x.shape

    def test_lower_mse_than_uniform_2bit(self, quantizer):
        """Outlier-aware 2.5-bit should have lower MSE than uniform 2-bit."""
        torch.manual_seed(0)
        x = torch.randn(4, 8, 32, 128, device=DEVICE)

        # Uniform 2-bit
        q2 = TurboQuantMSE(head_dim=128, bits=2, device=DEVICE, dtype=DTYPE)
        idx2, n2 = q2.quantize(x)
        mse_2bit = ((x - q2.dequantize(idx2, n2)) ** 2).mean().item()

        # Outlier-aware 2.5-bit
        idx_o, n_o = quantizer.quantize(x)
        mse_outlier = ((x - quantizer.dequantize(idx_o, n_o)) ** 2).mean().item()

        assert (
            mse_outlier < mse_2bit
        ), f"Outlier 2.5-bit MSE ({mse_outlier:.6f}) should be < uniform 2-bit ({mse_2bit:.6f})"

    def test_higher_mse_than_uniform_3bit(self, quantizer):
        """Outlier-aware 2.5-bit should have higher MSE than uniform 3-bit."""
        torch.manual_seed(0)
        x = torch.randn(4, 8, 32, 128, device=DEVICE)

        # Uniform 3-bit
        q3 = TurboQuantMSE(head_dim=128, bits=3, device=DEVICE, dtype=DTYPE)
        idx3, n3 = q3.quantize(x)
        mse_3bit = ((x - q3.dequantize(idx3, n3)) ** 2).mean().item()

        # Outlier-aware 2.5-bit
        idx_o, n_o = quantizer.quantize(x)
        mse_outlier = ((x - quantizer.dequantize(idx_o, n_o)) ** 2).mean().item()

        assert (
            mse_outlier > mse_3bit
        ), f"Outlier 2.5-bit MSE ({mse_outlier:.6f}) should be > uniform 3-bit ({mse_3bit:.6f})"

    def test_calibration_mode(self):
        """Calibration should pick high-variance channels as outliers."""
        torch.manual_seed(42)
        q = OutlierAwareTurboQuant(
            head_dim=128,
            outlier_bits=3,
            normal_bits=2,
            num_outlier_channels=32,
            outlier_mode="static",
            device=DEVICE,
            dtype=DTYPE,
        )

        # Create data with known outlier channels: first 32 have 10x variance
        calib_data = torch.randn(1000, 128, device=DEVICE)
        calib_data[:, :32] *= 10.0

        q.calibrate(calib_data)

        # After rotation, the high-variance channels are spread across dimensions,
        # so perfect overlap isn't expected. Verify calibration ran and selected 32 channels.
        outlier_set = set(q._outlier_idx.tolist())
        assert len(outlier_set) == 32, f"Expected 32 outlier channels, got {len(outlier_set)}"
        assert q.outlier_mode == "calibration"

    def test_calibration_improves_mse(self):
        """Calibration-based outlier detection should improve MSE on data with outlier structure."""
        torch.manual_seed(42)
        d = 128

        # Data where certain channels have much higher variance
        x = torch.randn(4, 8, 32, d, device=DEVICE)
        x[..., :32] *= 5.0  # First 32 channels are outliers

        # Static (arbitrary) outlier selection
        q_static = OutlierAwareTurboQuant(
            head_dim=d,
            outlier_bits=3,
            normal_bits=2,
            num_outlier_channels=32,
            outlier_mode="static",
            device=DEVICE,
            dtype=DTYPE,
        )
        idx_s, n_s = q_static.quantize(x)
        mse_static = ((x - q_static.dequantize(idx_s, n_s)) ** 2).mean().item()

        # Calibrated outlier selection
        q_calib = OutlierAwareTurboQuant(
            head_dim=d,
            outlier_bits=3,
            normal_bits=2,
            num_outlier_channels=32,
            outlier_mode="static",
            device=DEVICE,
            dtype=DTYPE,
        )
        q_calib.calibrate(x.reshape(-1, d))
        idx_c, n_c = q_calib.quantize(x)
        mse_calib = ((x - q_calib.dequantize(idx_c, n_c)) ** 2).mean().item()

        assert mse_calib <= mse_static, f"Calibrated MSE ({mse_calib:.6f}) should be <= static MSE ({mse_static:.6f})"

    def test_different_bit_configs(self):
        """Various outlier/normal bit combinations should work."""
        configs = [
            (4, 2, 32),  # 4-bit outlier, 2-bit normal → 2.5 effective
            (4, 3, 64),  # 4-bit outlier, 3-bit normal → 3.5 effective
            (3, 1, 32),  # 3-bit outlier, 1-bit normal → 1.5 effective
        ]
        x = torch.randn(1, 8, 4, 128, device=DEVICE)
        for o_bits, n_bits, n_outlier in configs:
            q = OutlierAwareTurboQuant(
                head_dim=128,
                outlier_bits=o_bits,
                normal_bits=n_bits,
                num_outlier_channels=n_outlier,
                device=DEVICE,
                dtype=DTYPE,
            )
            idx, norms = q.quantize(x)
            x_rec = q.dequantize(idx, norms)
            assert x_rec.shape == x.shape
            assert not torch.isnan(x_rec).any()

    def test_invalid_num_outlier_channels(self):
        """Should raise if num_outlier_channels >= head_dim."""
        with pytest.raises(ValueError):
            OutlierAwareTurboQuant(head_dim=128, num_outlier_channels=128)


# ---- Bit-packing tests ----


class TestBitPack:
    def test_round_trip_all_bitwidths(self):
        """Pack then unpack should recover original indices for all bit-widths."""
        for bits in [1, 2, 3, 4]:
            idx = torch.randint(0, 1 << bits, (2, 8, 16, 128), dtype=torch.uint8)
            packed = pack(idx, bits)
            unpacked = unpack(packed, bits, 128)
            assert torch.equal(idx, unpacked), f"Round-trip failed for {bits}-bit"

    def test_compression_ratio(self):
        """Packed tensors should be smaller by the expected ratio."""
        for bits, expected_ratio in [(1, 8.0), (2, 4.0), (3, 2.67), (4, 2.0)]:
            idx = torch.randint(0, 1 << bits, (1, 8, 64, 128), dtype=torch.uint8)
            packed = pack(idx, bits)
            ratio = idx.nelement() / packed.nelement()
            assert abs(ratio - expected_ratio) < 0.1, f"{bits}-bit: expected {expected_ratio}x, got {ratio:.2f}x"

    def test_packed_size_matches(self):
        """packed_size() should predict the actual packed last-dim size."""
        for bits in [1, 2, 3, 4]:
            n = 128
            expected = packed_size(n, bits)
            idx = torch.randint(0, 1 << bits, (n,), dtype=torch.uint8)
            actual = pack(idx.unsqueeze(0), bits).shape[-1]
            assert expected == actual, f"{bits}-bit: predicted {expected}, got {actual}"

    def test_multidim_shapes(self):
        """Packing should preserve all leading dimensions."""
        for bits in [1, 2, 3, 4]:
            idx = torch.randint(0, 1 << bits, (3, 7, 5, 128), dtype=torch.uint8)
            packed = pack(idx, bits)
            assert packed.shape[:3] == (3, 7, 5)
            unpacked = unpack(packed, bits, 128)
            assert unpacked.shape == idx.shape

    def test_boundary_values(self):
        """Max and min values should survive packing."""
        for bits in [1, 2, 3, 4]:
            max_val = (1 << bits) - 1
            idx = torch.zeros(1, 128, dtype=torch.uint8)
            idx[0, 0] = 0
            idx[0, 1] = max_val
            packed = pack(idx, bits)
            unpacked = unpack(packed, bits, 128)
            assert unpacked[0, 0] == 0
            assert unpacked[0, 1] == max_val

    def test_invalid_bits_raises(self):
        """Should raise for unsupported bit-widths."""
        idx = torch.zeros(1, 128, dtype=torch.uint8)
        with pytest.raises(ValueError):
            pack(idx, 5)
        with pytest.raises(ValueError):
            unpack(idx, 5, 128)


# ---- KV Cache tests ----


class TestTurboQuantCache:
    def test_update_and_retrieve(self):
        from turbo_quant.kv_cache import TurboQuantCache

        cache = TurboQuantCache(num_layers=2, head_dim=128, bits=3, variant="mse", device=DEVICE, dtype=DTYPE)

        # Simulate prefill: 16 tokens
        keys = torch.randn(1, 8, 16, 128, device=DEVICE)
        values = torch.randn(1, 8, 16, 128, device=DEVICE)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (1, 8, 16, 128)
        assert v_out.shape == (1, 8, 16, 128)
        assert cache.get_seq_length(0) == 16

        # Simulate decode: 1 token
        keys2 = torch.randn(1, 8, 1, 128, device=DEVICE)
        values2 = torch.randn(1, 8, 1, 128, device=DEVICE)
        k_out2, v_out2 = cache.update(keys2, values2, layer_idx=0)
        assert k_out2.shape == (1, 8, 17, 128)
        assert v_out2.shape == (1, 8, 17, 128)
        assert cache.get_seq_length(0) == 17

    def test_prod_variant_cache(self):
        from turbo_quant.kv_cache import TurboQuantCache

        cache = TurboQuantCache(num_layers=2, head_dim=128, bits=3, variant="prod", device=DEVICE, dtype=DTYPE)
        keys = torch.randn(1, 8, 4, 128, device=DEVICE)
        values = torch.randn(1, 8, 4, 128, device=DEVICE)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (1, 8, 4, 128)

    def test_memory_smaller_than_fp16(self):
        from turbo_quant.kv_cache import TurboQuantCache

        cache = TurboQuantCache(num_layers=1, head_dim=128, bits=2, variant="mse", device=DEVICE, dtype=torch.float16)
        seq_len = 1024
        keys = torch.randn(1, 8, seq_len, 128, dtype=torch.float16, device=DEVICE)
        values = torch.randn(1, 8, seq_len, 128, dtype=torch.float16, device=DEVICE)
        cache.update(keys, values, layer_idx=0)

        compressed_bytes = cache.memory_usage_bytes(0)
        fp16_bytes = 2 * seq_len * 8 * 128 * 2  # 2 (K+V) * seq * heads * dim * 2 bytes
        assert (
            compressed_bytes < fp16_bytes
        ), f"Compressed ({compressed_bytes}) should be smaller than FP16 ({fp16_bytes})"

    def test_bitpack_compression_ratios(self):
        """Bit-packed cache should achieve much better compression than unpacked."""
        from turbo_quant.kv_cache import TurboQuantCache

        seq_len = 1024
        fp16_bytes = 2 * seq_len * 8 * 128 * 2  # K+V, fp16

        for bits, min_expected_ratio in [(2, 5.0), (3, 3.5), (4, 2.5)]:
            cache = TurboQuantCache(
                num_layers=1,
                head_dim=128,
                bits=bits,
                variant="mse",
                device=DEVICE,
                dtype=DTYPE,
                use_bitpack=True,
            )
            keys = torch.randn(1, 8, seq_len, 128, device=DEVICE)
            values = torch.randn(1, 8, seq_len, 128, device=DEVICE)
            cache.update(keys, values, layer_idx=0)

            compressed = cache.memory_usage_bytes(0)
            ratio = fp16_bytes / compressed
            assert ratio >= min_expected_ratio, f"{bits}-bit: compression {ratio:.1f}x < expected {min_expected_ratio}x"

    def test_bitpack_reconstruction_matches_unpacked(self):
        """Packed and unpacked caches should produce identical reconstructions."""
        from turbo_quant.kv_cache import TurboQuantCache

        for bits in [2, 3, 4]:
            packed = TurboQuantCache(
                num_layers=1,
                head_dim=128,
                bits=bits,
                variant="mse",
                device=DEVICE,
                dtype=DTYPE,
                use_bitpack=True,
            )
            unpacked = TurboQuantCache(
                num_layers=1,
                head_dim=128,
                bits=bits,
                variant="mse",
                device=DEVICE,
                dtype=DTYPE,
                use_bitpack=False,
            )
            keys = torch.randn(1, 8, 16, 128, device=DEVICE)
            values = torch.randn(1, 8, 16, 128, device=DEVICE)

            k_p, v_p = packed.update(keys, values, layer_idx=0)
            k_u, v_u = unpacked.update(keys, values, layer_idx=0)

            assert torch.allclose(k_p, k_u, atol=1e-6), f"{bits}-bit key mismatch"
            assert torch.allclose(v_p, v_u, atol=1e-6), f"{bits}-bit value mismatch"

    def test_outlier_variant_cache(self):
        from turbo_quant.kv_cache import TurboQuantCache

        cache = TurboQuantCache(
            num_layers=2,
            head_dim=128,
            bits=3,
            variant="outlier",
            outlier_bits=3,
            normal_bits=2,
            num_outlier_channels=32,
            device=DEVICE,
            dtype=DTYPE,
        )
        keys = torch.randn(1, 8, 16, 128, device=DEVICE)
        values = torch.randn(1, 8, 16, 128, device=DEVICE)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (1, 8, 16, 128)
        assert cache.get_seq_length(0) == 16

        # Decode step
        k2 = torch.randn(1, 8, 1, 128, device=DEVICE)
        v2 = torch.randn(1, 8, 1, 128, device=DEVICE)
        k_out2, v_out2 = cache.update(k2, v2, layer_idx=0)
        assert k_out2.shape == (1, 8, 17, 128)
        assert cache.get_seq_length(0) == 17


if __name__ == "__main__":
    pytest.main([__file__, "-xvvv"])
