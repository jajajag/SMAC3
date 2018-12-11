from fabric import Connection
from absl import app
from fabric import SerialGroup

# 配置文件放在这里
nodes = [{
    'ssh_host': '52.80.187.152',
    # 这个文件就不上传了
    'ssh_key': 'smac/pssmac/utils/id_rsa_zhanghanping',
    'ssh_user': 'zhanghanping',
    'ssh_port': 22,
    'smac_dir': '$HOME/SMAC3/',
}, {
    'ssh_host': '54.223.203.214',
    'ssh_key': 'smac/pssmac/utils/id_rsa_zhanghanping',
    'ssh_user': 'zhanghanping',
    'ssh_port': 22,
    'smac_dir': '$HOME/SMAC3/',
}, {
    'ssh_host': '52.80.220.75',
    'ssh_key': 'smac/pssmac/utils/id_rsa_zhanghanping',
    'ssh_user': 'zhanghanping',
    'ssh_port': 22,
    'smac_dir': '$HOME/SMAC3/',
}]


def main(argv):
    """Main test process for pssmac.

    Parameters
    ----------
    argv : Useless

    """
    # 创建分组
    group = SerialGroup(nodes[0]['ssh_host'],
                        nodes[0]['ssh_host'],
                        nodes[0]['ssh_host'],
                        connect_kwargs={'key_filename': nodes[0]['ssh_key']},
                        user=nodes[0]['ssh_user'],
                        port=nodes[0]['ssh_port'])
    # 执行脚本
    group.run(nodes[0]['smac_dir'] + 'test.sh')


if __name__ == '__main__':
    app.run(main)
