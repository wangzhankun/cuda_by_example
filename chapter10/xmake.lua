target("basic_double_stream_correct")
	set_kind("binary")
	add_files("basic_double_stream_correct.cu")
target_end()


target("basic_single_stream")
	set_kind("binary")
	add_files("basic_single_stream.cu")
target_end()


target("basic_double_stream")
	set_kind("binary")
	add_files("basic_double_stream.cu")
target_end()


target("copy_timed")
	set_kind("binary")
	add_files("copy_timed.cu")
target_end()


