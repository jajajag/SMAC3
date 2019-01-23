#include <boost/algorithm/string.hpp>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "ps.h"

// 定义key数组长度为502，其中一个为incumbent，剩下500个challengers，还有一个存数据大小和超参个数
#define KEY_SIZE 500 + 2

using Val = float;
using Key = ps::Key;

struct MyVal {
    // 什么都不存，空着
    // 如果能有文档说明怎么调用每个key对应的MyVal，就不用空着了
    //std::vector <Val> configHistory;

    inline void Load(dmlc::Stream *fi) {/* fi->Read(&w); */}

    inline void Save(dmlc::Stream *fo) const {/* fo->Write(w); */}

    inline bool Empty() const { return false; }
};

/* Read the data from Python.
 * 示例：
 * 2(行数) 1(超参个数)
 * 1.0(超参) 2(history数) (0.88 1.23 1111) (0.74 0.9 2222) (分别是cost, time, seed)
 * 0.8(超参) 1(history数) (0.5 1.5 3333)
 * */
void ReadPython(std::vector <std::vector<Val>> &my_val) {
    // 清空每个vector中的元素
    for (std::vector <Val> &vec : my_val)
        vec.clear();

    // 为了统一c++和python端的代码，我们按行读取，而不是按数据了
    // 会显得繁琐一些
    std::string line;
    int num_lines, num_config;
    float time_left;
    while (getline(std::cin, line)) {
        // 调用boost库的trim函数，去除string中的空字符
        boost::trim(line);
        if (line.empty())
            continue;
        std::stringstream stream(line);
        // 首先读入超参组的数量和超参个数，存入第一个vector中
        stream >> num_lines >> num_config >> time_left;
        my_val[0].push_back(static_cast<Val>(num_lines));
        my_val[0].push_back(static_cast<Val>(num_config));
        break;
    }

    int line_count = 1;
    while (line_count <= num_lines) {
        // 调用boost库的trim函数，去除string中的空字符
        getline(std::cin, line);
        boost::trim(line);
        if (line.empty())
            continue;
        // 有内容，则用字符串流读入
        std::stringstream stream(line);

        // 首先，读取一行中的每个超参
        Val val;
        for (int j = 0; j < num_config; ++j) {
            stream >> val;
            my_val[line_count].push_back(val);
        }

        // 其次，读取runhistory的个数，存放
        int num_runhistory;
        stream >> num_runhistory;
        my_val[line_count].push_back(static_cast<Val>(num_runhistory));

        // 最后，读入每个runhistory
        Val cost, time, seed;
        for (int j = 0; j < num_runhistory; ++j) {
            stream >> cost >> time >> seed;
            my_val[line_count].push_back(cost);
            my_val[line_count].push_back(time);
            my_val[line_count].push_back(seed);
        }
        // 有行读入，自加
        ++line_count;
    }
}

/* Write the data to Python. */
void WritePython(std::vector <std::vector<Val>> &my_val) {
    // 也统一使用字符串流输出
    std::stringstream stream;
    // 需要防止seed输出为科学计数法
    stream.setf(std::ios::fixed, std::ios::floatfield);
    // 首先输出超参组数和超参个数
    stream << my_val[0][0] << " " << my_val[0][1] << " ";
    stream << my_val[0][2];
    // 输出字符串以及行分隔符
    std::cout << stream.str() << std::endl;

    int num_lines = static_cast<int>(my_val[0][0]);
    // 输出每行的内容
    for (int i = 1; i <= num_lines; ++i) {
        std::stringstream stream;
        // 需要防止seed输出为科学计数法
        stream.setf(std::ios::fixed, std::ios::floatfield);
        // 输出所有vector存的数据
        for (const Val &config : my_val[i])
            stream << config << " ";
        // 输出字符串以及行分隔符
        std::cout << stream.str() << std::endl;
    }
}

void MyValToVector(const std::vector <std::vector<Val>> &my_val,
                   std::vector <Val> &val, std::vector<int> &siz) {
    // 首先清空两个数组
    val.clear();
    siz.clear();
    for (const std::vector <Val> &vec : my_val) {
        // 将每个vector连接到val的后面
        val.insert(val.end(), vec.begin(), vec.end());
        // 将每个size放到siz里
        siz.push_back(vec.size());
    }
}

void VectorToMyVal(std::vector <std::vector<Val>> &my_val,
                   const std::vector <Val> &val, const std::vector<int> &siz) {
    // 清空每个vector中的元素
    for (std::vector <Val> &vec : my_val)
        vec.clear();

    // 将每个siz对应的vec加到my_val里面
    for (size_t i = 0, pos = 0; i < siz.size(); ++i) {
        // 按理说用size_t最好，但既然他全用的int，我也用int
        my_val[i].insert(my_val[i].begin(), val.begin() + pos,
                         val.begin() + pos + siz[i]);
        pos += siz[i];
    }
}

class MyHandle {
public:
    /* Execute before the first push/pull operation. */
    void Start(bool push, int timestamp, int cmd, void *msg) {
        // 设置push的flag，如果worker端pull，则从python端读取数据
        this->push = push;
        this->my_val.resize(KEY_SIZE);
        if (!this->push)
            ReadPython(this->my_val);
    }

    /* Execute after the last push/pull operation. */
    void Finish() {
        // 如果worker端进行push，则将结果写入python端
        if (this->push)
            WritePython(this->my_val);
    }

    /* Handle a single push by key. */
    void Push(Key recv_key, ps::Blob<const Val> recv_val, MyVal &my_val) {
        // 将类内对应的key的ConfigHistory覆盖
        this->my_val[recv_key].assign(recv_val.begin(), recv_val.end());
    }

    /* Handle a single pull by key. */
    void Pull(Key recv_key, MyVal &my_val, ps::Blob <Val> &send_val) {
        // 调用key对应的vector，data和size分别对应数据和长度
        send_val.data = this->my_val[recv_key].data();
        send_val.size = this->my_val[recv_key].size();
    }

    // 不调用这些嘻嘻哈哈的函数
    inline void Load(dmlc::Stream *fi) {}

    inline void Save(dmlc::Stream *fo) const {}

private:
    // 可以将ts_设置为timestamp，但没必要。
    //int ts_ = 0;
    bool push;

    // 中间结果保存在这里
    std::vector <std::vector<Val>> my_val;

};

int CreateServerNode(int argc, char *argv[]) {
    using Server = ps::OnlineServer<Val, MyVal, MyHandle>;
    Server server;
    return 0;
}

int WorkerNodeMain(int argc, char *argv[]) {
    using namespace ps;
    KVWorker <Val> wk;
    // val和siz中存放数据
    std::vector <Val> val;
    std::vector<int> siz;
    std::vector <std::vector<Val>> my_val(KEY_SIZE);
    // 将key的值设定为[0, KEY_SIZE]，方便取值
    std::vector <Key> key(KEY_SIZE);
    for (Key i = 0; i < KEY_SIZE; ++i)
        key[i] = i;

    // 传输的主过程
    while (true) {
        // 从python端读取数据，存到my_val，再进行转化
        ReadPython(my_val);
        MyValToVector(my_val, val, siz);
        // push完用同个数组接收
        wk.Wait(wk.VPush(key, val, siz));
        wk.Wait(wk.VPull(key, &val, &siz));
        // 写入到python端
        VectorToMyVal(my_val, val, siz);
        WritePython(my_val);
    }

    return 0;
}
