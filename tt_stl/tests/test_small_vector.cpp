// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//
//
//                     SmallVector Unit Tests
//
// This file contains a comprehensive suite of unit tests for a
// `SmallVector<T, N>` implementation.  The tests cover construction and
// destruction semantics, behaviour of the small‐buffer optimisation (SBO),
// element access and iteration, modifying operations, capacity queries,
// exception safety guarantees and a handful of edge cases.  Each test is
// documented with a brief comment describing its intent.
//
// A `SmallVector` stores up to `N` elements in an internal buffer located
// inside the object itself.  When more than `N` elements are stored the
// container falls back to heap allocation.  This optimisation avoids heap
// allocations for the common case where only a few elements are stored
// The internal data pointer initially points at the small buffer.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <tt_stl/small_vector.hpp>

namespace tt::stl {
namespace {

constexpr std::size_t kInlineCapacity = 4;

struct Large {
    int data[256];
    static int dtorCount;
    Large() = default;
    Large(const Large&) = default;
    Large& operator=(const Large&) = default;
    ~Large() { ++dtorCount; }
};

int Large::dtorCount = 0;

// A helper type that counts constructor and destructor calls.  It is used
// throughout the tests to verify that operations call constructors and
// destructors the expected number of times.  Each test must reset the
// static counters before use.
struct Tracked {
    static int ctorCount;
    static int copyCtorCount;
    static int moveCtorCount;
    static int dtorCount;
    int value;
    Tracked(int v = 0) : value(v) { ++ctorCount; }
    Tracked(const Tracked& other) : value(other.value) { ++copyCtorCount; }
    Tracked(Tracked&& other) noexcept : value(other.value) {
        ++moveCtorCount;
        other.value = -1;
    }
    Tracked& operator=(const Tracked& other) = default;
    Tracked& operator=(Tracked&& other) noexcept = default;
    ~Tracked() { ++dtorCount; }
    friend bool operator==(const Tracked& lhs, const Tracked& rhs) { return lhs.value == rhs.value; }
};

int Tracked::ctorCount = 0;
int Tracked::copyCtorCount = 0;
int Tracked::moveCtorCount = 0;
int Tracked::dtorCount = 0;

// Over‐aligned type used to verify that SmallVector honours alignment
// requirements.  OverAlignment of 64 bytes is sufficient to exercise
// alignment handling on most platforms.
struct alignas(64) OverAligned {
    int payload;
    OverAligned(int v = 0) : payload(v) {}
    friend bool operator==(const OverAligned& lhs, const OverAligned& rhs) { return lhs.payload == rhs.payload; }
};

// Fixture for tests using `SmallVector<int, kInlineCapacity>`.  Provides convenience
// typedefs and ensures the vector type is available via `Vec`.
class SmallVectorIntTest : public ::testing::Test {
protected:
    using Vec = SmallVector<int, kInlineCapacity>;
};

// Fixture for tests using `SmallVector<Tracked, 4>`.  Resets the static
// counters in `Tracked` before each test to ensure accurate counting.
class SmallVectorTrackedTest : public ::testing::Test {
protected:
    using Vec = SmallVector<Tracked, kInlineCapacity>;
    void SetUp() override {
        Tracked::ctorCount = 0;
        Tracked::copyCtorCount = 0;
        Tracked::moveCtorCount = 0;
        Tracked::dtorCount = 0;
    }
    void TearDown() override {
        // No specific action needed; destructors should have been counted.
    }
};

//===----------------------------------------------------------------------===//
// Construction & Destruction
//
// Tests verifying that `SmallVector` can be default constructed, constructed
// from ranges or initializer lists, and copied or moved correctly.  We also
// verify that destructors run the correct number of times when the vector
// goes out of scope.

TEST_F(SmallVectorIntTest, DefaultConstructionIsEmpty) {
    // A default constructed vector should have zero size and be empty.  The
    // capacity should be at least the statically defined N (kInlineCapacity in this case).
    Vec vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_GE(vec.capacity(), kInlineCapacity);
}

TEST_F(SmallVectorIntTest, ConstructionFromIteratorRange) {
    // Constructing from a pair of iterators should populate the vector with
    // the values in that range.
    int arr[] = {1, 2, 3, 4};
    Vec vec(std::begin(arr), std::end(arr));
    EXPECT_EQ(vec.size(), 4u);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(vec[i], arr[i]);
    }
}

TEST_F(SmallVectorIntTest, InitializerListConstruction) {
    // An initializer list constructor should take the list of values and store
    // them in order.
    Vec vec{5, 6, 7};
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 5);
    EXPECT_EQ(vec[1], 6);
    EXPECT_EQ(vec[2], 7);
}

TEST_F(SmallVectorTrackedTest, CopyConstructorSmallAndLarge) {
    // Copying a vector that fits in the small buffer should result in another
    // vector using its own small buffer and containing the same elements.
    Vec small;
    small.emplace_back(1);
    small.emplace_back(2);
    small.emplace_back(3);
    const auto smallData = small.data();
    Vec copySmall(small);
    EXPECT_EQ(copySmall.size(), small.size());
    for (std::size_t i = 0; i < copySmall.size(); ++i) {
        EXPECT_EQ(copySmall[i].value, small[i].value);
    }
    // The copied vector should have its own storage; addresses must differ.
    EXPECT_NE(copySmall.data(), smallData);
    // Now create a vector that requires heap storage and copy it.
    Vec large;
    // Push more than the inline capacity to force heap allocation.
    for (int i = 0; i < kInlineCapacity + 4; ++i) {
        large.emplace_back(i);
    }
    const auto largeData = large.data();
    EXPECT_GT(large.capacity(), kInlineCapacity);
    Vec copyLarge(large);
    EXPECT_EQ(copyLarge.size(), large.size());
    for (std::size_t i = 0; i < copyLarge.size(); ++i) {
        EXPECT_EQ(copyLarge[i].value, large[i].value);
    }
    // The copy should not alias the original heap buffer.
    EXPECT_NE(copyLarge.data(), largeData);
    EXPECT_GE(copyLarge.capacity(), large.size());
}

TEST_F(SmallVectorTrackedTest, MoveConstructorSmallAndLarge) {
    // Moving a vector should transfer its contents.  After the move the
    // destination should contain the original elements and the source should
    // be in a valid but unspecified state (size zero is acceptable).  For
    // small vectors the destination's data pointer will point at its own
    // small buffer; for large vectors it may take ownership of the heap.
    {
        Vec small;
        small.emplace_back(10);
        small.emplace_back(20);
        auto moved = std::move(small);
        EXPECT_EQ(moved.size(), 2u);
        EXPECT_EQ(moved[0].value, 10);
        EXPECT_EQ(moved[1].value, 20);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_TRUE(small.empty());
    }
    {
        Vec large;
        for (int i = 0; i < kInlineCapacity + 2; ++i) {
            large.emplace_back(i);
        }
        // Force allocation beyond inline capacity
        const auto oldData = large.data();
        const std::size_t oldCapacity = large.capacity();
        Vec moved = std::move(large);
        EXPECT_EQ(moved.size(), kInlineCapacity + 2);
        EXPECT_GE(moved.capacity(), oldCapacity);
        // When moved, it is typical for the heap storage to be transferred.
        EXPECT_EQ(moved.data(), oldData);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_TRUE(large.empty());
        // The moved-from vector may have shrunk its capacity but should be
        // functional; pushing into it should work.
        large.emplace_back(42);
        EXPECT_EQ(large.size(), 1u);
        EXPECT_EQ(large[0].value, 42);
    }
}

TEST_F(SmallVectorTrackedTest, DestructorCalls) {
    {
        Vec vec;
        for (int i = 0; i < kInlineCapacity; ++i) {
            vec.emplace_back(i);
        }
        EXPECT_EQ(Tracked::dtorCount, 0);
        // When the vector goes out of scope all contained objects must be
        // destroyed exactly once.
    }
    EXPECT_EQ(Tracked::dtorCount, kInlineCapacity);
}

//===----------------------------------------------------------------------===//
// Small‐Buffer Behaviour
//
// The following tests verify that the vector uses the inlined buffer for up
// to `N` elements and correctly transitions to heap storage when necessary.

TEST_F(SmallVectorIntTest, PushWithinSmallBufferKeepsInplaceStorage) {
    Vec vec;
    // Push elements one at a time; after the first push record the pointer
    // returned by data().  Subsequent pushes up to the inline capacity
    // should not change this pointer.
    vec.push_back(0);
    auto initialPtr = vec.data();
    auto initialCap = vec.capacity();
    for (int i = 1; i < kInlineCapacity; ++i) {
        vec.push_back(i);
        EXPECT_EQ(vec.data(), initialPtr);
        EXPECT_EQ(vec.capacity(), initialCap);
    }
    EXPECT_EQ(vec.size(), kInlineCapacity);
}

TEST_F(SmallVectorIntTest, PushBeyondSmallBufferTriggersHeapAllocation) {
    Vec vec;
    for (int i = 0; i < kInlineCapacity; ++i) {
        vec.push_back(i);
    }
    const auto smallPtr = vec.data();
    const auto smallCap = vec.capacity();
    // Pushing the fifth element should trigger a reallocation to the heap.
    vec.push_back(4);
    EXPECT_GT(vec.capacity(), smallCap);
    EXPECT_NE(vec.data(), smallPtr);
    EXPECT_EQ(vec[4], 4);
}

//===----------------------------------------------------------------------===//
// Element Access & Iterators
//

TEST_F(SmallVectorIntTest, FrontAndBackAccessors) {
    Vec vec{1, 2, 3};
    EXPECT_EQ(vec.front(), 1);
    EXPECT_EQ(vec.back(), 3);
    // Mutate the back element and verify the change.
    vec.back() = 5;
    EXPECT_EQ(vec.back(), 5);
    EXPECT_EQ(vec[2], 5);
}

TEST_F(SmallVectorIntTest, ForwardAndReverseIterators) {
    Vec vec{1, 2, 3, 4};
    // Sum elements via forward iterators.
    int sum = 0;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 10);
    // Copy into a std::vector via const iterators.
    const Vec& cvec = vec;
    std::vector<int> copy(cvec.cbegin(), cvec.cend());
    EXPECT_EQ(copy.size(), vec.size());
    EXPECT_TRUE(std::equal(copy.begin(), copy.end(), vec.begin()));
    // Reverse iteration should traverse the elements backwards.
    std::vector<int> reversed(vec.rbegin(), vec.rend());
    EXPECT_EQ(reversed.size(), vec.size());
    EXPECT_EQ(reversed[0], 4);
    EXPECT_EQ(reversed[1], 3);
    EXPECT_EQ(reversed[2], 2);
    EXPECT_EQ(reversed[3], 1);
    // Const reverse iterators.
    std::vector<int> crev(cvec.crbegin(), cvec.crend());
    EXPECT_EQ(crev, reversed);
}

TEST_F(SmallVectorIntTest, ConstVsNonConstIteratorTypes) {
    Vec vec{1, 2, 3};
    // The type of *cbegin should be const int& and not modifiable.
    using ConstRef = decltype(*vec.cbegin());
    static_assert(std::is_same_v<ConstRef, const int&>, "cbegin() must return const ref");
    // The type of *begin should be int&.
    using Ref = decltype(*vec.begin());
    static_assert(std::is_same_v<Ref, int&>, "begin() must return ref");
    // Ensure iterators cover all elements.
    size_t count = 0;
    for (auto&& x : vec) {
        (void)x;
        ++count;
    }
    EXPECT_EQ(count, vec.size());
}

//===----------------------------------------------------------------------===//
// Modifiers
//
// Tests that exercise push_back, emplace_back, pop_back, insert, erase, clear,
// swap, resize, reserve and shrink_to_fit.  Each operation is checked for
// correctness of the resulting contents, size/capacity and element lifetime.

TEST_F(SmallVectorIntTest, emplaceBackWithArray) {
    // Original implementation didn't support emplace_back for std::array, which led to compilation error.
    // This test verifies that it works now.
    SmallVector<std::array<int, 3>, kInlineCapacity> vec;
    vec.emplace_back(7, 8, 9);
    ASSERT_EQ(vec.size(), 1u);
}

TEST_F(SmallVectorTrackedTest, PushPopEmplaceBack) {
    Vec vec;
    // Emplace constructs in place; constructor count will increase.
    vec.emplace_back(1);
    EXPECT_EQ(Tracked::ctorCount, 1);
    vec.emplace_back(2);
    vec.emplace_back(3);
    EXPECT_EQ(vec.size(), 3u);
    // Pop should call one destructor.
    vec.pop_back();
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(Tracked::dtorCount, 1);
    // Pushing back a copy should invoke the copy constructor.
    Tracked t(42);
    vec.push_back(t);
    EXPECT_EQ(vec.back().value, 42);
    EXPECT_EQ(Tracked::copyCtorCount, 1);
}

TEST_F(SmallVectorIntTest, InsertAtVariousPositions) {
    Vec vec{1, 4};
    // Insert a single element in the middle.
    auto it = vec.insert(vec.begin() + 1, 2);
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(*it, 2);
    // Insert a range at the end.  This may trigger a growth.
    std::initializer_list<int> tail = {5, 6};
    vec.insert(vec.end(), tail.begin(), tail.end());
    EXPECT_EQ(vec.size(), 5u);
    EXPECT_EQ(vec[3], 5);
    EXPECT_EQ(vec[4], 6);
    // Insert at the beginning.
    vec.insert(vec.begin(), 0);
    EXPECT_EQ(vec.front(), 0);
    EXPECT_EQ(vec.size(), 6u);

    std::vector<int> expected{0, 1, 2, 4, 5, 6};
    EXPECT_TRUE(std::equal(vec.begin(), vec.end(), expected.begin()));
}

TEST_F(SmallVectorIntTest, EraseSingleAndRange) {
    Vec vec{0, 1, 2, 3, 4};
    // Erase a single element.
    auto it = vec.erase(vec.begin() + 1);
    EXPECT_EQ(*it, 2);
    EXPECT_EQ(vec.size(), 4u);
    std::vector<int> expected1{0, 2, 3, 4};
    EXPECT_TRUE(std::equal(vec.begin(), vec.end(), expected1.begin()));
    // Erase a range in the middle.
    it = vec.erase(vec.begin() + 1, vec.begin() + 3);
    EXPECT_EQ(*it, 4);
    EXPECT_EQ(vec.size(), 2u);
    std::vector<int> expected2{0, 4};
    EXPECT_TRUE(std::equal(vec.begin(), vec.end(), expected2.begin()));
}

TEST_F(SmallVectorIntTest, SwapExchangesContentsAndStorage) {
    Vec small{1, 2, 3};
    Vec large;
    // Force large to allocate on the heap by pushing beyond inline capacity.
    for (int i = 0; i < kInlineCapacity + 2; ++i) {
        large.push_back(i + 10);
    }
    std::size_t largeCap = large.capacity();

    std::size_t small_size = small.size();
    std::size_t large_size = large.size();

    small.swap(large);

    // After the swap the data pointers and capacities should also have swapped.
    EXPECT_EQ(small.size(), large_size);
    EXPECT_EQ(large.size(), small_size);
    EXPECT_EQ(small.capacity(), largeCap);
    // Capacity won't shrink after swap
    EXPECT_EQ(large.capacity(), largeCap);

    // Element values should be preserved.
    EXPECT_EQ(small[0], 10);
    EXPECT_EQ(small[5], 15);
    EXPECT_EQ(large[0], 1);
    EXPECT_EQ(large[2], 3);
}

TEST_F(SmallVectorTrackedTest, ResizeAndReserve) {
    Vec vec;
    // Increase size; default constructed values have value 0.
    vec.resize(3);
    EXPECT_EQ(vec.size(), 3u);
    for (const auto& e : vec) {
        EXPECT_EQ(e.value, 0);
    }
    // Resize down; destructors should be called.
    vec.resize(1);
    EXPECT_EQ(vec.size(), 1u);
    EXPECT_EQ(Tracked::dtorCount, 2);
    // Reserve additional capacity; this should not change the size.
    auto oldCap = vec.capacity();
    vec.reserve(oldCap + 10);
    EXPECT_GE(vec.capacity(), oldCap + 10);
    EXPECT_EQ(vec.size(), 1u);
    // Clearing should remove elements but not deallocate.
    vec.clear();
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_GE(vec.capacity(), oldCap + 10);
}

//===----------------------------------------------------------------------===//
// Capacity Queries
//
// Simple checks of size(), capacity() and empty() in both small and large
// configurations.

TEST_F(SmallVectorIntTest, SizeCapacityAndEmpty) {
    Vec vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_GE(vec.capacity(), kInlineCapacity);
    vec.push_back(1);
    EXPECT_FALSE(vec.empty());
    EXPECT_EQ(vec.size(), 1u);
    auto smallCap = vec.capacity();
    // Fill up to small buffer capacity.
    for (int i = 1; i < kInlineCapacity; ++i) {
        vec.push_back(i + 1);
    }
    EXPECT_EQ(vec.size(), kInlineCapacity);
    EXPECT_EQ(vec.capacity(), smallCap);
    // Now exceed the small buffer.
    vec.push_back(5);
    EXPECT_EQ(vec.size(), 5u);
    EXPECT_GT(vec.capacity(), smallCap);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//
// Tests for zero inline capacity, large element types and alignment
// requirements.

TEST(SmallVectorEdgeCaseTest, ZeroCapacityUsesHeapAlways) {
    // When N == 0 there is no inline storage; the vector must allocate from
    // the heap for every element.
    SmallVector<int, 0> vec;
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), 0u);
    // Push a few elements and observe capacity growth.
    vec.push_back(1);
    EXPECT_GE(vec.capacity(), 1u);

    //  current capacity growth algorithm is  2 * OldCapacity + 1.
    vec.push_back(2);
    // Pushing another element may reallocate; capacity should be >= 2
    EXPECT_GE(vec.capacity(), 2u);
}

TEST(SmallVectorEdgeCaseTest, LargeTypesWithNonTrivialDestructors) {
    // Use a large array type as the element to ensure the implementation can
    // handle large sizes.  The destructor should be called for each element.

    Large::dtorCount = 0;
    {
        SmallVector<Large, 2> vec;
        vec.resize(5);
        EXPECT_EQ(vec.size(), 5u);
    }
    // Five destructors should have been called when vec went out of scope.
    EXPECT_EQ(Large::dtorCount, 5);
}

TEST(SmallVectorEdgeCaseTest, OverAlignedTypesAreProperlyAllocated) {
    // Over‐aligned types should still have their alignment honoured in the
    // vector's storage.  We check the returned data pointer alignment.
    SmallVector<OverAligned, 2> vec;
    vec.emplace_back(7);
    std::uintptr_t ptrVal = reinterpret_cast<std::uintptr_t>(vec.data());
    EXPECT_EQ(ptrVal % alignof(OverAligned), 0u);
    vec.emplace_back(9);

#ifdef HEAP_MEMORY_ALIGNMENT_IS_FIXED
    // TODO: Heap memory is not properly aligned; this also applies to boost::container::small_vector.
    // Growing beyond inline capacity may move to heap storage; alignment
    // requirements should still be satisfied.
    vec.emplace_back(11);
    ptrVal = reinterpret_cast<std::uintptr_t>(vec.data());
    EXPECT_EQ(ptrVal % alignof(OverAligned), 0u);
    EXPECT_EQ(vec[2].payload, 11);
#endif
    // Verify contents.
    EXPECT_EQ(vec[0].payload, 7);
    EXPECT_EQ(vec[1].payload, 9);
}

}  // namespace
}  // namespace tt::stl
