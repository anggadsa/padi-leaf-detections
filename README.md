# Paddy Leaf Disease Detection
Bangkit's Capstone Project : Paddy Leaf Disease Detection
## Paddy Leaf Disease Detection Team
* Machine Learning Path
	* Syachrul Qolbi Nur Septi https://www.linkedin.com/in/syachrulqolbi/
	* Fadhilah Nur Rismayana https://www.linkedin.com/in/fadhilah-nur-rismayana/
* Mobile Development
	* Selvia Indriani https://www.linkedin.com/in/selviaindriani/
	* Muhammad Rifki https://www.linkedin.com/in/muhammad-rifki/
* Cloud Computing
    * Gustiana Nurhadi https://www.linkedin.com/in/gustiana-nurhadi-231aa8121/
	* Angga Dwi Satria https://www.linkedin.com/in/anggadwisatria/

## Steps to Configure Architecture and Deployment Model with Flask 
* A. Create 2 buckets in cloud storage for GET and POSH processes resulting from the detection process
* B. 
   *  1. Create a virtual machine on the GCP compute engine with Deep Learning for Linux Tensorflow 2.2 image
   *  2. Install libraries in virtual machines such as
      a. tensorflow==2.2.0
      b. h5py==2.10.0
      c. numpy==1.19.5
      d. flask==2.0.1
      e. tensorflow-hub==0.12.0
   *  3. main.py Scripting Process
   *  4. Run main.py with the command sudo python3 main.py& (on linux)
   *  5. The flash is running and the API has been obtained according to the External IP on the compute engine
