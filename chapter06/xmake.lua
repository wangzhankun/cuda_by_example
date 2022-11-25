target("ray")
	set_kind("binary")
	add_files("ray.cu")
target_end()


target("ray_noconst")
	set_kind("binary")
	add_files("ray_noconst.cu")
target_end()


