# AutoEncoder_vehicle.
This repo contain all the materiel needed to reproduce the results of our method/paper `Enhanced Autoencoder-based LiDAR Localization in Self-Driving Vehicles`.


# Instal libraries.
`!pip install pandas`
`!pip install matplotlib`
`!pip install tqdm`
`!pip install pykitti`
`!pip install opencv-python`

Certainly! Here's an example README section to explain the dataset configuration for your GitHub repository, incorporating the code snippet you provided:

---

## Dataset Configuration

Setting up the dataset for this project is a crucial step. To streamline this process, we've prepared the `main.ipynb` notebook that helps facilitate the necessary configurations.

The primary cell in `main.ipynb` handles the dataset setup and prepares it for training. Here's an excerpt from the notebook:

```python
file='/home/anas/Desktop/PHD/my essai/Kitti dataset/City' #[long (2011_10_03,27) ,Residence, Campus-8, Person-8, Road, City]
seque = ['00','05','06','07','08','09','10']
seq = 13
date =['2011_10_03', '2011_09_30', '2011_09_30', '2011_09_30', '2011_09_30','2011_09_30', '2011_09_30']
drive=['0027', '0018','0020', '0027', '0028', '0033', '0034']

# For NCLT data. Try to specify the path
path = '/home/anas/Desktop/PHD/my essai/NCLT dataset'
date1 = ['2013-04-05','2013-01-10', '2012-12-01', '2012-01-08', '2012-06-15', '2012-04-29', '2012-02-04']
```

Ensure to modify the variables accordingly:

- `file`: Specify the path to your dataset.
- `pathfile`: For NCLT data, specify the NCLT dataset path.
- `seque`: List of sequences.
- `seq`: Sequence number.
- `date`: List of dates.
- `drive`: List of drives.

Utilize the `main.ipynb` notebook to manage and configure the dataset for your specific use case.

--- 

If everything goes well then yoou will be able to train the model and generate the results.

If you encounter any problem, you can send me a mail to 
- `anas.charroud@usmba.ac.ma`
