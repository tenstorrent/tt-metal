import csv
import flatbuffers
import ttnn.Reshard
import ttnn.ReshardTable


builder = flatbuffers.Builder(1024)

with open("0.csv") as f:
    serialized_rows = []
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        in_grid_str = builder.CreateString(row["in_grid"])
        out_grid_str = builder.CreateString(row["out_grid"])
        group_grid_str = builder.CreateString(row["group"])

        ttnn.Reshard.ReshardStart(builder)
        ttnn.Reshard.ReshardAddInGrid(builder, in_grid_str)
        ttnn.Reshard.ReshardAddOutGrid(builder, out_grid_str)
        ttnn.Reshard.ReshardAddGroup(builder, group_grid_str)
        ttnn.Reshard.ReshardAddCoef(builder, float(row["coef"]))
        ttnn.Reshard.ReshardAddIntercept(builder, float(row["intercept"]))
        ttnn.Reshard.ReshardAddRSq(builder, float(row["r^2"]))
        ttnn.Reshard.ReshardAddRrmse(builder, float(row["RRMSE"]))
        ttnn.Reshard.ReshardAddRmsre(builder, float(row["RMSRE"]))
        ttnn.Reshard.ReshardAddNumPoints(builder, int(row["num_points"]))
        row_data = ttnn.Reshard.ReshardEnd(builder)

        serialized_rows.append(row_data)

    ttnn.ReshardTable.ReshardTableStartRowsVector(builder, len(serialized_rows))
    for row in serialized_rows:
        builder.PrependUOffsetTRelative(row)
    rows_vector = builder.EndVector()

    ttnn.ReshardTable.ReshardTableStart(builder)
    ttnn.ReshardTable.ReshardTableAddRows(builder, rows_vector)
    reshard_table = ttnn.ReshardTable.ReshardTableEnd(builder)

    builder.Finish(reshard_table)

    buf = builder.Output()
    with open("v.bin", "wb") as fout:
        fout.write(buf)
