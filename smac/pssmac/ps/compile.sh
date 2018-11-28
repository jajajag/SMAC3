example_name=$1
g++ -std=c++11 -msse2 -fPIC -Wall -finline-functions -O3 -ggdb -I $PS_LITE/src -I $PS_LITE/deps/include    -fopenmp $example_name $PS_LITE/build/libps.a $PS_LITE/build/libps_main.a $PS_LITE/deps/lib/libprotobuf.a $PS_LITE/deps/lib/libglog.a $PS_LITE/deps/lib/libgflags.a $PS_LITE/deps/lib/libzmq.a $PS_LITE/deps/lib/libcityhash.a $PS_LITE/deps/lib/liblz4.a -lpthread  -lrt -o ${example_name:0:-3}
# 需要把c++代码编译了，设置PS_LITE环境变量为ps-lite文件夹路径，不带/
