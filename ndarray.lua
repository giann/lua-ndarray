function table.map(t, mapFn)
    local mapped = {}

    for k, v in pairs(t) do
        table.insert(mapped, mapFn(k, v))
    end

    return mapped
end

function table.imap(t, mapFn)
    local mapped = {}

    for i, v in ipairs(t) do
        table.insert(mapped, mapFn(i, v))
    end

    return mapped
end

local function iota(n)
    local result = {}
    for i = 1, n do
        result[i] = i
    end
    return result
end

local function order(self)
    local stride = self.stride
    local terms = {}
    for i = 1, #stride do
        terms[i] = {math.abs(stride[i]), i}
    end

    table.sort(terms, function(a, b)
        return a[1] < b[1]
    end)

    local result = {}
    for i = 1, #terms do
        result[i] = terms[i][2]
    end

    return result
end

local constructorCache = {}

local function compileConstructor(dimension)
    local indices = iota(dimension)
    local args = table.imap(indices, function(i, _)
        return "i" .. i
    end)
    local index_str = "self.offset + " .. table.concat(table.imap(indices, function(i, _)
        return "self.stride[" .. i .. "] * i" .. i
    end), "+")
    local shapeArg = table.concat(
        table.imap(indices, function(i, _)
            return "b" .. i
        end),
        ","
    )
    local strideArg = table.concat(
        table.imap(indices, function(i, _)
            return "c" .. i
        end),
        ","
    )

    local name = "ndarray" .. dimension .. "d"

    local code = "local " .. name
        .. "\n" .. name.. " = function (CTOR_LIST, ORDER, a, "
        .. shapeArg .. ","
        .. strideArg .. ", d)"
        .. "\nlocal ndarray = {}"
        .. "\nndarray.data=a"
        .. "\nndarray.shape = {" .. shapeArg .. "}"
        .. "\nndarray.stride = {" .. strideArg .. "}"
        .. "\nndarray.offset = math.floor(d)"
        .. "\nndarray.dimension = " .. dimension

        -- size
        .. "\nndarray.size = function(self)"
        .. "\n    return " .. table.concat(
                table.imap(indices, function(i, _)
                    return "ndarray.shape[" .. i .. "]"
                end),
                "*"
            )
        .. "\nend"

    -- order
    if dimension == 1 then
        code = code
            .. "\nndarray.order = {0}"
    else
        code = code
            .. "\nndarray.order = "

        if dimension < 4 then
            code = code
                .. "\nfunction(self)"

            if dimension == 2 then
                code = code
                    .. "\nreturn math.abs(self.stride[1]) > math.abs(self.stride[2]) and {1,0} or {0,1}"
                    .. "\nend"
            elseif dimension == 3 then
                code = code
                    .. "\nlocal s0 = math.abs(self.stride[1])"
                    .. "\nlocal s1 = math.abs(self.stride[2])"
                    .. "\nlocal s2 = math.abs(self.stride[3])"
                    .. "\nif s0 > s1 then"
                    .. "\n    if s1 > s2 then"
                    .. "\n        return {2,1,0}"
                    .. "\n    elseif s0 > s2 then"
                    .. "\n        return {1,0,2}"
                    .. "\n    end"
                    .. "\nelseif s0 > s2 then"
                    .. "\n    return {2,0,1}"
                    .. "\nelseif s2 > s1 then"
                    .. "\n    return {0,1,2}"
                    .. "\nelse"
                    .. "\n    return {0,2,1}"
                    .. "\nend"
                    .. "\nend"
            end
        else
            code = code
                .. "ORDER"
        end
    end

    -- set
    code = code
        .. "\nndarray.set = function(self," .. table.concat(args, ",") .. ",v)"
        .. "\n    self.data[" .. index_str .. "] = v"
        .. "\nend"

    -- get
    code = code
        .. "\nndarray.get = function(self," .. table.concat(args, ",") .. ")"
        .. "\n    return self.data[" .. index_str .. "]"
        .. "\nend"

    -- index
    code = code
        .. "\nndarray.index = function(self," .. table.concat(args, ",") .. ")"
        .. "\n    return " .. index_str
        .. "\nend"

    -- hi
    code = code
        .. "\nndarray.hi = function(self," .. table.concat(args, ",") .. ")"
        .. "\n    return " .. name .. "(CTOR_LIST, ORDER, self.data,"
        .. "\n        "
        .. table.concat(
            table.imap(indices, function (i, _)
                return "type(i" .. i .. ") ~= \"number\" or i" .. i .. " < 0"
                    .. " and self.shape[" .. i .. "] or math.floor(i" .. i .. ")"
            end),
            ","
        ) .. ","
        .. table.concat(
            table.imap(indices, function (i, _)
                return "self.stride[" .. i .. "]"
            end),
            ","
        ) .. ", self.offset)"
        .. "\nend"

    -- lo
    local a_vars = table.imap(indices, function(i, _)
        return "\nlocal a" .. i .. " = self.shape[" .. i .. "]"
    end)
    local c_vars = table.imap(indices, function(i, _)
        return "\nlocal c" .. i .. " = self.stride[" .. i .. "]"
    end)

    code = code
        .. "\nndarray.lo = function(self," .. table.concat(args, ",") .. ")"
        .. "\n    local b = self.offset"
        .. "\n    local d = 0"
        .. "\n    " .. table.concat(a_vars, "") .. table.concat(c_vars, "")

    for i = 1, dimension do
        code = code
            .. "\nif type(i" .. i .. ") == \"number\" and i" .. i .. " >= 1 then"
            .. "\n    d = math.floor(i" .. i .. ")"
            .. "\n    b = b + c" .. i .. " * d"
            .. "\n    a" .. i .. " = a" .. i .. " - d"
            .. "\nend"
    end

    code = code
        .. "\nreturn " .. name .. "(CTOR_LIST, ORDER, self.data,"
        .. table.concat(
            table.imap(indices, function(i, _)
                return "a" .. i
            end)
            , ","
        ) .. ","
        .. table.concat(
            table.imap(indices, function(i, _)
                return "c" .. i
            end)
            , ","
        ) .. ", b)"
        .. "\nend"

    -- step
    code = code
        .. "\nndarray.step = function(self," .. table.concat(args, ",") .. ")"
        .. "\n" .. table.concat(
            table.imap(indices, function(i, _)
                return "local a" .. i .. " = self.shape[" .. i .. "]"
            end),
            "\n"
        )
        .. table.concat(
            table.imap(indices, function(i, _)
                return "local b" .. i .. " = self.stride[" .. i .. "]"
            end),
            "\n"
        )
        .. "\nlocal c = self.soffset"
        .. "\nlocal d = 1"

    for i = 1, dimension do
        code = code
            .. "\nif type(i" .. i .. ") == \"number\" then"
            .. "\n    d = math.floor(i" .. i .. ")"
            .. "\n    if d < 1 then"
            .. "\n        c = c + b" .. i .. " * (a" .. i .. " - 1)"
            .. "\n        a" .. i .. " = math.ceil(-a" .. i .. " / d)"
            .. "\n    else"
            .. "\n        a" .. i .. " = math.ceil(a" .. i .. " / d)"
            .. "\n    end"
            .. "\n    b" .. i .. " = b" .. i .. " * d"
            .. "\nend"
    end

    code = code
        .. "\nreturn " .. name .. "(CTOR_LIST, ORDER, self.data"
        .. table.concat(
            table.imap(indices, function(i, _)
                return "a" .. i
            end),
            ","
        ) .. ","
        .. table.concat(
            table.imap(indices, function(i, _)
                return "b" .. i
            end),
            ","
        ) .. ",c)"
        .. "\nend"

    -- transpose
    local tShape = {}
    local tStride = {}
    for i = 1, dimension do
        tShape[i] = "a[i" .. i .. "]"
        tStride[i] = "b[i" .. i .. "]"
    end
    code = code
        .. "\nndarray.transpose = function(self," .. table.concat(args, ",") .. ")"
        .. "\n    " .. table.concat(
            table.imap(
                args,
                function(idx, n)
                    return n .. " = not " .. n .. " and " .. idx .. " or math.floor(" .. n .. ")"
                end
            ),
            "\n"
        )
        .. "\n    local a = self.shape"
        .. "\n    local b = self.stride"
        .. "\n    return " .. name
        .. "(CTOR_LIST, ORDER, self.data, " .. table.concat(tShape, ",")
        .. "," .. table.concat(tStride, ",") .. ",self.offset)"
        .. "\nend"

    -- pick
    code = code
        .. "\nndarray.pick = function (self," .. table.concat(args, ",") .. ")"
        .. "\n    local a = {}"
        .. "\n    local b = {}"
        .. "\n    local c = self.offset"

    for i = 1, dimension do
        code = code
            .. "\n   if type(i" .. i ..") == \"number\" and i" .. i .. " >= 1 then"
            .. "\n       c = math.floor(c + self.stride[" .. i .. "] * i" .. i .. ")"
            .. "\n   else"
            .. "\n       table.insert(a, self.shape[" .. i .. "])"
            .. "\n       table.insert(b, self.stride[" .. i .. "])"
            .. "\n   end"
    end

    code = code
        .. "\n    local ctor = CTOR_LIST[#a]"
        .. "\n    return ctor(self.data, a, b, c)"
        .. "\nend"
        .. "\nreturn ndarray"
        .. "\nend"

    code = code
        .. "\nreturn function(CTOR_LIST, ORDER, data, shape, stride, offset)"
        .. "\n   return " .. name .. "(CTOR_LIST, ORDER, data,"
        .. table.concat(
            table.imap(indices, function(i, _)
                return "shape[" .. i .. "]"
            end),
            ","
        ) .. ","
        .. table.concat(
            table.imap(indices, function(i, _)
                return "stride[" .. i .. "]"
            end),
            ","
        ) .. ",offset)"
        .. "\nend"

    return (loadstring or load)(code)()
end

return function(data, shape, stride, offset)
    assert(shape and #shape > 0, "dimension must be at least 1")

    shape = shape or { #data }

    if not stride then
        stride = {}
        local sz = 1
        for i = #shape, 1, -1 do
            stride[i] = sz
            sz = sz * shape[i]
        end
    end

    if not offset then
        offset = 1
        for i = 1, #shape do
            if stride[i] < 1 then
                offset = offset - shape[i] * stride[i]
            end
        end
    end

    while #constructorCache < #shape do
        table.insert(constructorCache, compileConstructor(#constructorCache + 1))
    end

    return constructorCache[#shape](constructorCache, order, data, shape, stride, offset)
end
