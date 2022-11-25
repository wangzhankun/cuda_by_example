target("simple_kernel_params")
	set_kind("binary")
	add_files("simple_kernel_params.cu")
target_end()


target("simple_device_call")
	set_kind("binary")
	add_files("simple_device_call.cu")
target_end()


target("enum_gpu")
	set_kind("binary")
	add_files("enum_gpu.cu")
target_end()


target("set_gpu")
	set_kind("binary")
	add_files("set_gpu.cu")
target_end()


target("simple_kernel")
	set_kind("binary")
	add_files("simple_kernel.cu")
target_end()


target("hello_world")
	set_kind("binary")
	add_files("hello_world.cu")
target_end()


