std = "luajit"

self = false --Ignore unused self warnings

ignore = {
    "212" --Unused argument.
}

globals = {
    "table.imap",
    "table.map"
}
