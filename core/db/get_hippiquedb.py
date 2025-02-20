from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def main(url, extract_to='data/'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    main('https://drive.mattgautier.fr/d/s/129NSVwKZGrit6fiZBzukMCZvIeChvQu/webapi/entry.cgi/hippique.db.zip?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A868421216719912258%22%5D&force_download=true&json_error=true&c2_offload=%22allow%22&_dc=1739472238469&sharing_token=%220DzsGpTLaGUOkYjcJ5Kfzsz9mNze.CDG.Hfn3RPy_P2IHTFCUlQVNpWFNttgkQXt7obflFBUDPihVXzrtwSa17oZq7e91K9e0ljGYO5Ybt6u.PGyiyQ600sF94h0U5FzhPawY45bTo5rgfKbqhEW5Nl8QM50NWi_PDUiMCxFfSjymXyDNaQMTKxZGWXfWcwVC.DbBPb5vDqXf8tjxDj8qJX3nVK_DkbqwF9S.heG70cBlB7F1nnYEI13%22')