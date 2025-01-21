from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def main(url, extract_to='data/'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    main('https://drive.mattgautier.fr/d/s/11q523Ywv0MSHXA7xRRg423oath4ymcA/webapi/entry.cgi/hippique.db.zip?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A864207862843356651%22%5D&force_download=true&json_error=true&c2_offload=%22allow%22&_dc=1737462756458&sharing_token=%22p92C0lsxLRCgRcQupcoo.tefF6zTKr5QyLe.LHH5zB6EYK9AvKfOv4KFSGifTJNnby7Cor6eS0GwfV6FfDCOvbONtL6V9xXmw3S9pqvPCIowKo7K_VvPT3UL.e20FsjsnxXD6JpW4RT3Eba.R5nbnrxPZ4cWApsCSry2623lgb2W2KH6fBZgerFPDyLs8Zf7G4KgQu4paZTHkQxeOoeeUxikoy3N.cv8G3hRTAZGsDfCyFG5orcgdRmt%22')