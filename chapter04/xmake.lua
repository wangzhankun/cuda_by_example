target("add_loop_gpu")
	set_kind("binary")
	add_files("add_loop_gpu.cu")
target_end()


target("add_loop_gpu2")
	set_kind("binary")
	add_files("add_loop_gpu2.cu")
target_end()

target("julia_cpu")
	set_kind("binary")
	add_files("julia_cpu.cu")
target_end()


target("julia_gpu")
	set_kind("binary")
	add_files("julia_gpu.cu")
target_end()

target("julia_gpu2")
	set_kind("binary")
	add_files("julia_gpu2.cu")
target_end()


target("add_loop_long")
	set_kind("binary")
	add_files("add_loop_long.cu")
target_end()


target("add_loop_cpu")
	set_kind("binary")
	add_files("add_loop_cpu.cu")
target_end()


