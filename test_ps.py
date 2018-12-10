from fabric import Connection
from absl import app

# 配置文件放在这里
nodes = [{
    'ssh_host': '52.80.187.152',
    # 这个文件就不上传了
    'ssh_key': 'smac/pssmac/utils/id_rsa_zhanghanping',
    'ssh_user': 'zhanghanping',
    'ssh_port': 22,
    'smac_dir': '$HOME/SMAC3/',
    'data_dir': '$HOME/SMAC3/data/libsvm_real-sim',
    'temp_dir': '$HOME/SMAC3/tmp/',
    # 说出来你可能不信，直接调用python3，fabric没法识别
    'python': '/opt/python3/bin/python3',
    # 需要把这个目录创建出来
    'output_dir': '$HOME/SMAC3/output/',
    'node_host': '172.31.29.227',
    'node_port': [6666, 23333],
    'job': ['server', 'worker'],
    'id': [0, 0],
    'per_run_time_limit': [60, 60],
    'total_time_limit': [300, 300],
}, {
    'ssh_host': '54.223.203.214',
    'ssh_key': 'smac/pssmac/utils/id_rsa_zhanghanping',
    'ssh_user': 'zhanghanping',
    'ssh_port': 22,
    'smac_dir': '$HOME/SMAC3/',
    'data_dir': '$HOME/SMAC3/data/libsvm_real-sim',
    'temp_dir': '$HOME/SMAC3/tmp/',
    'python': '/opt/python3/bin/python3',
    'output_dir': '$HOME/SMAC3/output/',
    'node_host': '172.31.21.27',
    'node_port': [6666, 23333],
    'job': ['scheduler', 'worker'],
    'id': [0, 1],
    'per_run_time_limit': [60, 60],
    'total_time_limit': [300, 300],
}, {
    'ssh_host': '52.80.220.75',
    'ssh_key': 'smac/pssmac/utils/id_rsa_zhanghanping',
    'ssh_user': 'zhanghanping',
    'ssh_port': 22,
    'smac_dir': '$HOME/SMAC3/',
    'data_dir': '$HOME/SMAC3/data/libsvm_real-sim',
    'temp_dir': '$HOME/SMAC3/tmp/',
    'python': '/opt/python3/bin/python3',
    'output_dir': '$HOME/SMAC3/output/',
    'node_host': '172.31.28.149',
    'node_port': [23333],
    'job': ['worker'],
    'id': [2],
    'per_run_time_limit': [60, 60],
    'total_time_limit': [300, 300],
}]


def main(argv):
    """Main test process for pssmac.

    Parameters
    ----------
    argv : Useless

    """
    # 统计server和worker数量
    num_servers, num_workers = 0, 0
    for node in nodes:
        for i in range(len(node['job'])):
            if node['job'][i] == 'server':
                num_servers += 1
            elif node['job'][i] == 'worker':
                num_workers += 1
            elif node['job'][i] == 'scheduler':
                scheduler_host = node['node_host']
                scheduler_port = node['node_port'][i]
            else:
                raise ValueError("Please specify a valid job.")

    for node in nodes:
        # 建立fabric的连接
        connection = Connection(node['ssh_host'],
                                connect_kwargs={
                                    'key_filename': node['ssh_key']},
                                user=node['ssh_user'],
                                port=node['ssh_port'])
        # 所有flags大集合
        flags_dict = {}
        execute = node['smac_dir'] + 'run_facade.py'
        flags_dict['ps'] = node['smac_dir'] + 'smac/pssmac/ps/ps_smac'
        flags_dict['data_dir'] = node['data_dir']
        flags_dict['temp_dir'] = node['temp_dir']
        flags_dict['num_servers'] = num_servers
        flags_dict['num_workers'] = num_workers
        flags_dict['scheduler_host'] = scheduler_host
        flags_dict['scheduler_port'] = scheduler_port
        flags_dict['node_host'] = node['node_host']

        for i in range(len(node['job'])):
            # 装填针对每个node的flag
            flags_dict['node'] = node['job'][i]
            flags_dict['node_port'] = node['node_port'][i]
            flags_dict['id'] = node['id'][i]
            # 总剩余的时间和单轮时间
            flags_dict['per_run_time_limit'] = node['per_run_time_limit'][i]
            flags_dict['total_time_limit'] = node['total_time_limit'][i]
            flags = ' '.join([('--' + key + '=' + str(val)) for (key, val) in
                              flags_dict.items()])
            # 创建output文件
            output_file = node['output_dir'] + node['job'][i] + str(
                node['id'][i]) + '.txt'
            connection.run('touch ' + output_file)
            # 如果nohup挂不上的话，可以考虑在结尾加上 sleep 1
            # 因为session可能会提前关闭
            flags = 'nohup ' + node['python'] + ' -u ' + execute + ' ' + flags \
                    + ' >& ' + output_file + ' &'
            print(flags)
            # 运行
            #connection.run(flags)
            # print(connection.run('hostname'))


if __name__ == '__main__':
    app.run(main)
