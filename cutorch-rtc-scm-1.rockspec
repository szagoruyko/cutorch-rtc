package = "cutorch-rtc"
version = "scm-1"

source = {
   url = "git://github.com/szagoruyko/cutorch-rtc.git",
}

description = {
   summary = "Torch7 FFI cutorch runtime compilation helpers",
   detailed = [[
   ]],
   homepage = "https://github.com/szagoruyko/cutorch-rtc",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nvrtc",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
]],
   install_command = "cd build && $(MAKE) install"
}
