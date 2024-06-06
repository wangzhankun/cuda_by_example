target("hist_gpu_shmem_atomics")
	set_kind("binary")
	add_files("hist_gpu_shmem_atomics.cu")
target_end()


target("hist_gpu_gmem_atomics")
	set_kind("binary")
	add_files("hist_gpu_gmem_atomics.cu")
target_end()


target("hist_cpu")
	set_kind("binary")
	add_files("hist_cpu.cu")
target_end()



target("hist_gpu_ch9")
	set_kind("binary")
	add_files("hist_gpu_ch9.cu")
target_end()


