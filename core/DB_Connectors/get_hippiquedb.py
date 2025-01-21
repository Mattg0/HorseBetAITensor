from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def main(url, extract_to='data/'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    main('https://drive.mattgautier.fr/d/s/11q9Bz3IDdGHFbFvwZr0L07ULldjavBn/webapi/entry.cgi/hippique.db.zip?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A864222513209588471%22%5D&force_download=true&json_error=true&c2_offload=%22allow%22&_dc=1737470344777&sharing_token=%22ZJK.4Ms9MF2zrTNDMsQpj7rW.ywTT68W_h37pdnQePF6gn11sUUW63ILscipTnxOpN7IbKMczWkvmv_L4NDEzVGMcQyNBATv_PE.ZBMKJatBqVcpx7qZKQxQ6sJAaVq098Zz199Xe6.PzgPlJzG1iuLzQhnQDYn5Z7wLlJpzvtriKWyd4rip0bYdcEGdCqpxAWon6TaEaINdrHp4E83d2eXt0OpQpewTBkMzH0.vcnf.Z124HmU5Bkc9%22')