target("hashtable_gpu")
	set_kind("binary")
	add_files("hashtable_gpu.cu")
target_end()


target("dot_appendix")
	set_kind("binary")
	add_files("dot.cu")
target_end()


target("hashtable_cpu")
	set_kind("binary")
	add_files("hashtable_cpu.cu")
target_end()


