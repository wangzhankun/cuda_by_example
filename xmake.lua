set_project("cuda_by_example")

add_rules("mode.debug", "mode.release")
set_languages("c++17")
add_requires("cmake::OpenCV", {system = true})

add_links("opencv_cudabgsegm",
      "opencv_cudaobjdetect",
      "opencv_cudastereo",
      "opencv_dnn",
      "opencv_highgui",
      "opencv_ml",
      "opencv_shape",
      "opencv_stitching",
      "opencv_cudafeatures2d",
      "opencv_superres",
      "opencv_cudacodec",
      "opencv_videostab",
      "opencv_cudaoptflow",
      "opencv_cudalegacy",
      "opencv_objdetect",
      "opencv_calib3d",
      "opencv_videoio",
      "opencv_photo",
      "opencv_imgcodecs",
      "opencv_features2d",
      "opencv_cudawarping",
      "opencv_cudaimgproc",
      "opencv_cudafilters",
      "opencv_video",
      "opencv_imgproc",
      "opencv_flann",
      "opencv_cudaarithm",
      "opencv_core",
      "opencv_cudev")

add_includedirs("common")

add_cuflags("--extended-lambda")

add_cugencodes("native")



includes("chapter03", 
        "chapter06", 
        "chapter04", 
        "chapter09",
        "chapter10",
        "chapter07",
        "chapter08",
        "chapter05",
        "chapter11",
        "appendix_a")


--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro defination
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

