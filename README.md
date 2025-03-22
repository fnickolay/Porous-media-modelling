# Porous-media-modelling
Python-based code for modelling light propagation in transparent porous media.

## Usage
Set porous media parameters:
```
SVMC_params = {'Rbins' : 10,                             # radius od a single pore
               'number_of_bubbles' : num_of_bubbles,     # number of pores
               'xyz_proportions' : [1, 1, deform],       # axis z deformation will be equal to deform variable
               'volume_original_shape' : [600, 600, 600] # shape of the media. only parallelepiped shape is avalible
}
```
Set parameters of SVMC-executive file:
```
modelling_params = ['-B', '______111111',
                    #'-B', '______100100',
                    '-H', f'{nphot}', # phot is the integer ammount of photons
                    '-f', 'merged_data_2.json', # specific file name for it to work. see library code for information
                    '-w', 'X']
```
Set materials properties:
```
materials = [{"mua":0, # outer matter
              "mus":0,
              "g":1,
              "n":1},

             {"mua":0., # media, index 1
              "mus":0.0,
              "g":0.95,
              "n":1.37},

             {"mua":0., # pores, index 2
              "mus":0.,
              "g":0.95, 
              "n":1},

             {"mua":100000, # absorbing layer, index 3
              "mus":0.0, 
              "g":0.95, 
              "n":1.37}]
```
Set modelling iteration ID for it to save properly:
```
_ID = '600_size'
```
Get the results of modelling iteration:
```
res, por = SVMC_experiment_iterarion(ID=_ID,
                                     SVMC_parameters=SVMC_params,
                                     Modelling_parameters=modelling_params,
                                     materials=materials, 
                                     nphotons=nphot)
```
Variable por stands for the porosity of resulting media. 
The returning res variable stores the information from the .jnii file, downloaded to the folder after the SVMC process been executed.
See official MCX Extreme documentation for detailed specifications.

## Acknowledgement
### This library contains MCX Extreme by Qianqian Fang for it's implementation:
### URL: https://github.com/fangq/mcx/
### License: GPL version 3 or later
