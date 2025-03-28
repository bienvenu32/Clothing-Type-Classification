import requests

def download_model():
    url = "https://drive.google.com/uc?id=1GV22U9uaXsKN8WGvzTosuqEBr02XMtzV"
    output = "clothing_model.h5"
    response = requests.get(url, stream=True)
    with open(output, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Model downloaded as {output}")

if __name__ == "__main__":
    download_model()
