sudo apt-get update
sudo apt-get install

#Python env setup
sudo apt install python3-pip -y
sudo apt install python3-autopep8 -y

pip3 install -r requirements.txt
pip3 install tensorflow==2.8.0 --no-cache-dir

#Docker installation
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu  $(lsb_release -cs)  stable"
sudo apt update
sudo apt-get install docker-ce -y
docker --version
sudo systemctl start docker
