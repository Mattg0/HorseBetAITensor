from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def main(url, extract_to='data/'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    main('https://drive.mattgautier.fr/d/s/11jOzohADSBLI1an9CqdyFCAZNZYdpic/webapi/entry.cgi/hippique.db.zip?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A862749787296937576%22%5D&force_download=true&json_error=true&c2_offload=%22allow%22&_dc=1736767677911&sharing_token=%22M5g6o37oXpblTN.Wck6fQmaIp3wdTMsjy6Z0BukxN4XEPmsY8iuibo51Yc146iQBn26XFe5b7n_yxSTZZeNITkk1pVvdZNQEeZH9LM1GHb6WDlKE8I_ixo5m0jrMOsNpdbLMKmRzcxzz5l3bFYQZrA.i2O2oPBSNhMk86xWt0r1QiHPBHRymekijAHQX8nGisY1AymrbWebzezxclTU4bP4eAVLxitH3CAn.z.wRKyW8c6BkIim6i4RY%22')