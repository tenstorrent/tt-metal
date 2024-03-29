# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sqlite3

import ttnn

DATABASE_FILE = ttnn.REPORTS_PATH / "ttnn.db"
SQLITE_CONNECTION = None


def get_or_create_sqlite_db(db_file):
    global SQLITE_CONNECTION

    if SQLITE_CONNECTION is not None:
        return SQLITE_CONNECTION

    if not db_file.parent.exists():
        db_file.parent.mkdir(parents=True)
    db_file.unlink(missing_ok=True)
    SQLITE_CONNECTION = sqlite3.connect(db_file)

    cursor = SQLITE_CONNECTION.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS operations
                (operation_id int, name text)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS buffer_pages
                (operation_id int, address int, device_id int, core_y int, core_x int, bank_id int, page_index int, page_address int, page_size int, buffer_type int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS devices
                (
                    device_id int,
                    l1_num_banks int,
                    l1_bank_size int,
                    num_y_compute_cores int,
                    num_x_compute_cores int,
                    address_at_first_l1_bank int,
                    address_at_first_l1_cb_buffer int,
                    num_banks_per_storage_core int,
                    num_compute_cores int,
                    num_storage_cores int,
                    total_l1_memory int,
                    total_l1_for_tensors int,
                    total_l1_for_interleaved_buffers int,
                    total_l1_for_sharded_buffers int,
                    cb_limit int
                )"""
    )
    SQLITE_CONNECTION.commit()
    return SQLITE_CONNECTION


DEVICE_IDS_IN_DATABASE = set()


def log(operation, operation_id, devices):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()
    cursor.execute(f"INSERT INTO operations VALUES ({operation_id}, '{operation.name}')")
    sqlite_connection.commit()

    for device in devices:
        if device.id() in DEVICE_IDS_IN_DATABASE:
            continue
        device_info = ttnn._ttnn.reports.get_device_info(device)
        cursor.execute(
            f"""INSERT INTO devices VALUES (
                {device.id()},
                {device_info.l1_num_banks},
                {device_info.l1_bank_size},
                {device_info.num_y_compute_cores},
                {device_info.num_x_compute_cores},
                {device_info.address_at_first_l1_bank},
                {device_info.address_at_first_l1_cb_buffer},
                {device_info.num_banks_per_storage_core},
                {device_info.num_compute_cores},
                {device_info.num_storage_cores},
                {device_info.total_l1_memory},
                {device_info.total_l1_for_tensors},
                {device_info.total_l1_for_interleaved_buffers},
                {device_info.total_l1_for_sharded_buffers},
                {device_info.cb_limit}
            )"""
        )
        sqlite_connection.commit()
        DEVICE_IDS_IN_DATABASE.add(device.id())

    if ttnn.ENABLE_BUFFER_REPORT:
        for buffer_page in ttnn._ttnn.reports.get_buffer_pages():
            cursor.execute(
                f"""INSERT INTO buffer_pages VALUES (
                    {operation_id},
                    {buffer_page.address},
                    {buffer_page.device_id},
                    {buffer_page.core_y},
                    {buffer_page.core_x},
                    {buffer_page.bank_id},
                    {buffer_page.page_index},
                    {buffer_page.page_address},
                    {buffer_page.page_size},
                    {buffer_page.buffer_type.value}
                )"""
            )
        sqlite_connection.commit()
