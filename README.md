from astropy.table import Table
import spectral_cube as sc
import astropy.units as u
import os
import numpy as np
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import glob

def process_source(sourcename, vlsr):
    x = vlsr-25
    y = vlsr+25
    rest_frequencies = {
        'CH3OCHO': [98.182336,98.190658,98.270501,98.278921,98.424207,98.431803,98.435802,98.606856,98.682615,98.712001,
                    98.747906,98.792289,99.133272,99.135762,100.294604,100.308179,100.482241,100.490682,100.681545,
                    ],
        'CH3OH': [97.678803,98.030648,100.638872],
        'C2H5CN': [98.523872,98.533987,98.544164,98.559927,98.566615,98.70107,99.681461,100.614281],
        'C2H5OH': [97.535908,98.230313,98.583898,98.983548,99.524091],
        'CH3OCH3': [99.32443,99.8364434],
        'CH3CHO': [98.8633135,100.1271643,100.6452294],
        'CH3COCH3': [98.8003978,99.0525592,99.256107,99.542604],
        }
    spw7_files = f'/Volumes/Untitled/band3/data/{sourcename}/{sourcename}.spw7_CH3OH.image.pbcor.fits'
    spw8_files = f'/Volumes/Untitled/band3/data/{sourcename}/{sourcename}.spw8_HC3N.image.pbcor.fits'
    original_spw7_file = sc.SpectralCube.read(spw7_files)
    original_spw8_file = sc.SpectralCube.read(spw8_files)
    original_spw7_file.allow_huge_operations = True
    original_spw8_file.allow_huge_operations = True
    original_spw7_cube = original_spw7_file.to(u.K)
    original_spw8_cube = original_spw8_file.to(u.K)
    # print(original_spw7_cube)
    # print(original_spw8_cube)
    for molecule, freq_list in rest_frequencies.items():
        for i, rest_freq in enumerate(freq_list, start=1):
            if rest_freq > 99.469796875:
                original_spw_cube = original_spw8_cube
            else:
                original_spw_cube = original_spw7_cube
            subcube = original_spw_cube.with_spectral_unit(u.km/u.s, velocity_convention='radio',
                                                        rest_value=rest_freq*u.GHz)
            subcube = subcube.spectral_slab((x)*u.km/u.s, (y)*u.km/u.s)
            # print('subcube.shape %i' %(subcube.shape))
            if not os.path.exists(f'/Volumes/Untitled/band3/data/{sourcename}/'):
                os.mkdir(f'/Volumes/Untitled/band3/data/{sourcename}/')
            if not os.path.exists(f'/Volumes/Untitled/band3/data/{sourcename}/stack/'):
                os.mkdir(f'/Volumes/Untitled/band3/data/{sourcename}/stack/')
            subcube.write(f'/Volumes/Untitled/band3/data/{sourcename}/stack/{sourcename}_{molecule}_{i}.fits', overwrite=True)
            del original_spw_cube
            del subcube       
    print('Cut data: Done.')


    n = 40
    new_velocity_axis = np.linspace(x, y, n)
    for molecule, freq_list in rest_frequencies.items():
        for i, freq in enumerate(freq_list, start=1):
            input_cube_filename = f'/Volumes/Untitled/band3/data/{sourcename}/stack/{sourcename}_{molecule}_{i}.fits'
            output_cube_filename = f'/Volumes/Untitled/band3/data/{sourcename}/stack/new_{sourcename}_{molecule}_{i}.fits'
            hdul = fits.open(input_cube_filename)
            cube_data = hdul[0].data
            velocity_axis = hdul[0].header['CRVAL3'] + (np.arange(hdul[0].header['NAXIS3'])- hdul[0].header['CRPIX3']) * hdul[0].header['CDELT3']
            if len(velocity_axis)<3:
                continue
            f = interp1d(velocity_axis, cube_data, kind='linear', axis=0, fill_value=0,bounds_error=False)
            rebinned_cube_data = f(new_velocity_axis)
            hdul[0].data = rebinned_cube_data
            hdul[0].header['CRVAL3'] = new_velocity_axis[0]
            hdul[0].header['CRPIX3'] = 1
            hdul[0].header['NAXIS3'] = n
            hdul[0].header['CDELT3'] = (new_velocity_axis[-1] - new_velocity_axis[0]) / (n - 1)
            hdul.writeto(output_cube_filename, overwrite=True)
            # print(f'{input_cube_filename} to {output_cube_filename}')
            del cube_data
            del hdul
    # print(hdul[0].header['CDELT3'])
    # print(f'Vxmax-Vxmin:{y - x}')
    # print(f'Channels:{n - 1}')
    print('Cut data: Done.')


    molecule_weights = {
            'CH3OCHO': [0.25094,0.517829,0.278374,0.950221,0.668888,0.943512,0.737702,0.738865,0.811227,0.64972,0.734352,
                        0.73348,0.24307,0.174699,0.948459,0.795285,1.01443,1.13842,1.31329],
            'CH3OH': [0.223293,0.10268,3.87118],
            'C2H5CN': [3.16777,0.421993,0.688789,0.122714,1.593669,1.63292,1.75846,1.85897],
            'C2H5OH': [0.227239,0.31656,0.276962,0.298745,0.290472],
            'CH3OCH3': [1.38234,0.32189],
            'CH3CHO': [0.6655,0.102925,0.102915],
            'CH3COCH3': [0.292441,0.373566,0.159959,0.125312],
    } 
    for molecule, freq_list in rest_frequencies.items():
        thefirst = glob.glob(f'/Volumes/Untitled/band3/data/{sourcename}/stack/new_{sourcename}_{molecule}_*.fits')[0]
        stack_cube = np.zeros_like(fits.getdata(thefirst))
        total_weight = 0
        normalized_weight = []
        for i, freq in enumerate(freq_list, start=1):
            if not os.path.exists(f'/Volumes/Untitled/band3/data/{sourcename}/stack/new_{sourcename}_{molecule}_{i}.fits'):
                continue
            cube = fits.open(f'/Volumes/Untitled/band3/data/{sourcename}/stack/new_{sourcename}_{molecule}_{i}.fits')
            weights = molecule_weights[molecule]
            total_weight += sum(weights)
            normalized_weights = normalized_weight + [w / total_weight for w in weights]
            stack_cube += normalized_weights[i - 1]*cube[0].data
            del cube
        header = fits.getheader(thefirst)   
        stack_cube_hdu = fits.PrimaryHDU(stack_cube,header=header)
        stack_cube_hdu.writeto(f'/Volumes/Untitled/band3/data/{sourcename}/stack/new_{sourcename}_{molecule}_stack.fits',overwrite=True)
        del stack_cube
        del stack_cube_hdu 
    print('stacking data: Done.')

    for molecule, freq_list in rest_frequencies.items():
        for i, freq in enumerate(freq_list, start=1):
            original_file_path = f'/Volumes/Untitled/band3/data/{sourcename}/stack/{sourcename}_{molecule}_{i}.fits'
            new_file_path = f'/Volumes/Untitled/band3/data/{sourcename}/stack/new_{sourcename}_{molecule}_{i}.fits'
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            else:
                continue
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            else:
                continue
    print('Data cleanup: Done.')

#Read the table
table = Table.read('/Volumes/Untitled/band3/result.csv',format='csv')
#Process each source
for number in range(0,147):
    selected_rows = table[table['ID'] == number]
    if len(selected_rows) > 0:
        sourcename = selected_rows['Name'][0]
        vlsr = selected_rows['Vlsr'][0]
        print(f'Processing {sourcename}, vlsr: {vlsr} km/s.')
        process_source(sourcename, vlsr)   
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
