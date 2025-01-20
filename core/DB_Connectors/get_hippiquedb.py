from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def main(url, extract_to='data/'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    main('https://drive.mattgautier.fr/d/s/11pFe9eb2nd5oNEfd5aQ6EI3IcLA27rb/webapi/entry.cgi/hippique.db.zip?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A864026902887707846%22%5D&force_download=true&json_error=true&c2_offload=%22allow%22&_dc=1737376952541&sharing_token=%228Lmu18rN9QFyhS_69Iti6pwvDWGeZvFzOZy3zArosVbgwdf7suCv2N0F6h9Je2bxmt7h2vff4DE5CmjhW9JwMszp8GyLr6RGFYVCTjhV4_YYvo5JdSpRn1iyZONUqEjPYVf95tfcTH_17o_sXEOFi7N5oVtHVG01gdFkek06wP9SSL_AcE7QJkRK6KARr9PpD0zDCoE2bIKGTZmOeKb.gNmpQpq3FTqvAY0BsU0wQYRCTuuiJlOYCO3H%22')