from fabric import Connection
from absl import app

nodes = [{
    'host': '54.80.187.152',
    'key': './id_rsa_zhanghanping',
    'user': 'zhanghanping',
    'ssh_port': 22,
    'node_port': [6666, 23333],
    'job': ['server', 'worker'],
    'id': [0, 0],
    'cutoff': [0, 60]
    'smac': '$HOME/smac',
    'data_dir': '$HOME/data/libsvm_real-sim',
    'temp_dir': '$HOME/tmp/',
}, {
    'host': '52.80.187.152',
    'key': './id_rsa_zhanghanping',
    'user': 'zhanghanping',
    'ssh_port': 22,
    'node_port': [6666, 23333],
    'job': ['scheduler', 'worker'],
    'id': [0, 1],
    'cutoff': [0, 60]
    'smac': '$HOME/smac',
    'data_dir': '$HOME/data/libsvm_real-sim',
    'temp_dir': '$HOME/tmp/',
}, {
    'host': '52.80.220.75',
    'key': './id_rsa_zhanghanping',
    'user': 'zhanghanping',
    'ssh_port': 22,
    'node_port': [23333],
    'job': ['worker'],
    'id': [2],
    'cutoff': [60]
    'smac': '$HOME/smac',
    'data_dir': '$HOME/data/libsvm_real-sim',
    'temp_dir': '$HOME/tmp/',
}]

"""
nohup
python3
test_ps.py
--ps=./smac/pssmac/ps/ps_smac
--data_dir=./data/libsvm_real-sim
--node=worker
--temp_dir=./tmp/
--num_servers=1
--num_workers=2
--scheduler_host=172.31.21.27
--scheduler_port=6666
--node_host=172.31.29.227
--node_port=23333
--id=0
--cutoff=60
1 > output/worker0_output.txt
&

"""


def main(argv):
    # 所有flags大集合
    flags_list = ['execute', 'ps', 'data_dir', 'node', 'temp_dir',
                  'num_servers', 'num_workers', 'scheduler_host',
                  'scheduler_port', 'node_host', 'node_port', 'id', 'cutoff']
    num_servers, num_workers = 0, 0
    for node in nodes:
        for i in range(len(node['job'])):
            if node['job'][i] == 'server':
                num_servers += 1
            elif node['job'][i] == 'worker':
                num_workers += 1
            elif node['job'][i] == 'scheduler':
                scheduler_host = node['']
            else:
                raise KeyError("Please specify a valid job.")


    for node in nodes:
        # 建立fabric的连接
        connection = Connection(node['host'],
                                connect_kwargs={'key_filename': node['key']},
                                user=node['user'], port=node['ssh_port'])
        # cd到smac的目录下
        connection.run('cd ' + node['smac'])
        #
        flags_dict = {}
        execute = node['smac'] + '/pssmac/utils/run_facade.py'
        flags_dict['ps'] = node['smac'] + '/pssmac/ps/ps_smac'
        flags_dict['data_dir'] = node['data_dir']
        flags_dict['temp_dir'] = node['temp_dir']
        flags_dict['num_server'] =

        flags = ' '.join([('--' + key + '=' + str(val)) for (key, val) in
                          flags_dict.items()])


if __name__ == '__main__':
    app.run(main)
