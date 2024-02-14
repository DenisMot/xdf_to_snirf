# xdf_to_snirf
Translation of XDF files recorded with ARTINIS Starstim8 to SNIRF format. 

The xdf format is a general format for storing time series data (https://github.com/sccn/xdf).   
The snirf format is a format for storing nirs data (https://github.com/fNIRS/snirf).


## Usage (without Git)
1. Download as a zip file (button `< > Code`)
2. Extract the archive on your computer (e.g., in your `Download` folder). 
3. Move the extracted `xdf_to_snirf` folder to its intended location (e.g., in your `Documents/CodeProjects/` directory)
4. In VSCode:
    1. Open the `xdf_to_snirf` folder in a new window. 
    2. Open `main.ipynb` and click `Run all`
        * Running `main.ipynb` (without modifications) will convert the `*.xdf` files in the `data/reference` folder to `*.snirf` files in the `results` folder (it will create `results`). 

## Directory structure
The logic is to organize the data, code, and results in separate folders, all within the `xdf_to_snirf` folder. The structure is as follows:   

    xdf_to_snirf
       ├── readme.md     # description of the project 
       ├── data          # data used as input in the project  
       ├── results       # results of the conversion 
       ├── notebooks     # analyses with jupyter notebooks 
       └── main.ipynb    # entry point: to run all analyses

## Prerequisites
You need to have a Jupyter environment installed on your computer. 
For a minimal installation, refer to https://github.com/DenisMot/Python-for-HMS-Template.

## Convert multiple XDF files to SNIRF format in batch 
This is the main purpose of this project.
To convert your own `.xdf` files, you need to set your own `xdf_files` list, which is the list of xdf files to convert. 

### Simple and easy way 
The easiest way is to modify the call to `get_xdf_files_in_directory()` in `main.ipynb`. For example, if your `.xdf` files are located in `/Users/denismot/my_data`, you should modify it as follows:
```python
xdf_files = get_xdf_files_in_directory('/Users/denismot/my_data')
```

### Advanced way
If you want to convert `.xdf` files from several directories, you can set the `xdf_files` list manually. For example, if you want to convert the first two files in `/Users/denismot/my_data`, you can do as follows:
```python
xdf_files = get_xdf_files_in_directory('/Users/denismot/my_data')
xdf_files = xdf_files[:2]
```

If you want to convert files from different directories, you can do as follows:
```python
xdf_files = get_xdf_files_in_directory('/Users/denismot/my_data')
xdf_files += get_xdf_files_in_directory('/Users/denismot/my_other_data')
```
**NOTE:** The converted files will always be saved in the `results` folder. If the names of the xdf files from different directories are the same, the converted files will overwrite each other.
