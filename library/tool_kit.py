# импорт необходимых библиотек
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import pathlib
import json
import jdata
import sys
import seaborn as sns
import math
import tqdm
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.draw import ellipsoid
from pathlib import Path

# SVMC-раздел ##############################################################################################################################

def SVMC_json_base(ID,
                   volume_dict, 
                   materials, 
                   nphotons):
    """
    Создает основу JSON-файла для моделирования SVMC. Ряд параметров задан по умолчанию, ряд параметров нужно передать.
    
    ID: str
    Название сессии. На ее основе SVMC будет выдавать названия файлов.
    
    volume_dict: {x: int, y: int, z: int}
    Словарь с размерами среды моделирования.
    
    materials: SVMC materials list
    Список материалов для моделирования. Задаются в виде {"mua":0, "mus":0, "g":1, "n":1}. 
    Обязателен первый для материала пустоты за образцом.
    
    nphotons: int
    Число фотонов для моделирования.
    
    Возвращает словарь с ключами для SVMC-моделирования.
    
    """
    # пустой словарь - база
    jbase = dict()
    
    # параметры сессси
    jbase["Session"] = {"ID":            ID,
                        "DoMismatch":    1,
                        "DoSaveExit":    'true',
                        "LengthUnit":    1,
                        "DoAutoThread":  1,
                        "RNGSeed":       209466425,
                        "Photons":       nphotons}
    
    # временные промежутки между шагами
    jbase["Forward"] = {"T0":  0,
                        "T1":  5e-07,
                        "Dt":  5e-07}
    
    # свойства среды
    jbase["Domain"] = {"OriginType":   1,
                       "LengthUnit":   1,
                       "Media":        materials,
                       "Dim":          [volume_dict['x'],volume_dict['y'],volume_dict['z']],
                       "MediaFormat":  "svmc",
                       "VolumeFile":   ""}
    
    return jbase


def SVMC_source(jbase,
                position,
                direction,
                radius):
    """
    Создает источник в форме диска и добавляет в указанный словарь для моделирования.
    
    jbase: dict
    Словарь, который будет конвертирован в JSON-файл для SVMC моделирования.
    
    position: [z, y, x], float
    Положение источника.
    
    direction: [cos(gamma), cos(beta), cos(alpha), focus], float
    Направление излучения. Фокус может быть отрицательным.
    
    radius: float
    Радиус диска.
    
    Вовращает None
    
    """
    jbase['Optode'] = {"Source":{"Type":    "disk",
                                 "Pos":     position,
                                 "Dir":     direction,
                                 "Param1":  [radius, 0.0, 0.0, 0.0],
                                 "Param2":  [0.0, 0.0, 0.0, 0.0]
                                }
                      }  


def SVMC_model_and_shapes_merger(str_model, 
                                 str_shapes, 
                                 output_name):
    """
    Объединяет JSON-файлы с параметрами моделирования и фигурами.
    
    str_model: str
    Название файла с параметрами.
    
    str_shapes: str
    Название файла с фигурами.
    
    output_name: str
    Название итогового файла.
    
    Возвращает None.
    
    """
    # файл с моделью (там все кроме Shapes)
    with open(str_model) as f1:
        data1 = json.load(f1)
    # файл с фигурами
    with open(str_shapes) as f2:
        data2 = json.load(f2)

    # единый JSON-файл    
    merged_data = {}
    merged_data.update(data1)
    merged_data.update(data2)    
    
    # отгружаем в текущую папку по заданному имени
    with open(output_name, 'w') as outfile:
        json.dump(merged_data, outfile)


def SVMC_sample_porosity(shapes, pore_material_index=2):
    """
    Пористость образца.
    
    shapes: 3D array
    3D массив со значенями, заполненными для SVMC-моделирования.
    
    pore_material_index: int
    Индекс материала, который считается материалом внутренности поры.
    
    Возвращает значение пористости.
    
    """
    return np.sum(shapes == pore_material_index) / np.sum(shapes != -1)
        
        

def porous_sample_creator(Rbins, 
                          volume_dict, 
                          num_of_bubbles, 
                          x_deform=1, y_deform=1, z_deform=1):
    """
    Создает образец, в котором случайно равномерно заполняет порами-эллипсоидами с заданной деформацией.
    
    Rbins: int
    Радиус пор в бинах, не считая центрального.
    
    volume_dict: {x: int, y: int, z: int}
    Словарь с размерами среды моделирования. 
    
    num_of_bubbles: int
    Число пор
    
    x_deform, y_deform, z_deform: int
    Отношение полуосей к полному радиусу.
    
    """
    volume_dict = {'x':int(volume_dict['x']*x_deform),
                   'y':int(volume_dict['y']*y_deform),
                   'z':int(volume_dict['z']*z_deform),}
    mask = sphere_creator(Rbins = Rbins, 
                          x_deform = z_deform, 
                          y_deform = y_deform, 
                          z_deform = x_deform)
    x_coords = np.random.uniform(low = 0, 
                                 high = volume_dict['x'] - mask.shape[2],
                                 size = num_of_bubbles).astype(int)
    y_coords = np.random.uniform(low = 0, 
                                 high = volume_dict['y'] - mask.shape[1],
                                 size = num_of_bubbles).astype(int)
    z_coords = np.random.uniform(low = 0, 
                                 high = volume_dict['z'] - mask.shape[0],
                                 size = num_of_bubbles).astype(int)

    coords = np.concatenate((z_coords.reshape(-1, 1), 
                             y_coords.reshape(-1, 1), 
                             x_coords.reshape(-1, 1)), axis = 1).astype(int)
    volume = np.ones((volume_dict['z'], 
                      volume_dict['y'], 
                      volume_dict['x'])).astype(np.uint64)
    #volume[0, 0, 0] = 3
    volume[:, 0:2, :] = 3
    volume[:, :, 0:2] = 3
    #volume[-1:, :, :] = 3
    volume[:, -1:, :] = 3
    volume[:, :, -1:] = 3
    
    
    def filler(coord):
        volume_slice = volume[coord[0]:coord[0]+mask.shape[0],
                              coord[1]:coord[1]+mask.shape[1],
                              coord[2]:coord[2]+mask.shape[2]]
        volume[coord[0]:coord[0]+mask.shape[0],
               coord[1]:coord[1]+mask.shape[1],
               coord[2]:coord[2]+mask.shape[2]] = np.where((volume_slice==1) | (mask==2), mask, volume_slice)

                            
    np.apply_along_axis(filler, -1, coords)
    return volume


def SVMC_experiment_iterarion(ID,
                              SVMC_parameters,
                              Modelling_parameters,
                              materials, 
                              nphotons,
                              wd,
                              prog,
                              output):
    """
    Одна итерация моделирования SVMC.
    
    ID: str
    Название сессии ID. На ее основе SVMC будет выдавать названия файлов.
    
    parameters: {
                 'Rbins' :                  int
                 'number_of_bubbles' :      int
                 'xyz_proportions' :        [x_deform, y_deform, z_deform], float
                 'volume_original_shape' :  [x_size, y_size, z_size], int
                }
    Основные параметры моделирования.
    
    materials: SVMC materials list
    Список материалов для моделирования. Задаются в виде {"mua":0, "mus":0, "g":1, "n":1}. 
    Обязателен первый для материала пустоты за образцом.
    
    nphotons: int
    Число фотонов в моделировании.

    wd: string
    Рабочая директория.

    prog: string
    Директория расположения исполняемого файла.

    output: string
    Директория, в которой будут сохраняться промежуточные результаты исполнения кода.
    
    Возвращает результат моделирования и пористость.
    
    """ 
    print('Starting iteration.')
    start = time.time()
    Rbins =            SVMC_parameters['Rbins']
    num_of_bubbles =   SVMC_parameters['number_of_bubbles']
    deforms =     {'x':SVMC_parameters['xyz_proportions'][0], 
                   'y':SVMC_parameters['xyz_proportions'][1], 
                   'z':SVMC_parameters['xyz_proportions'][2]}
    volume_dict = {'x':SVMC_parameters['volume_original_shape'][0], 
                   'y':SVMC_parameters['volume_original_shape'][1], 
                   'z':SVMC_parameters['volume_original_shape'][2]}
    json_base = SVMC_json_base(ID=ID,
                               volume_dict=volume_dict, 
                               materials=materials, 
                               nphotons=nphotons)
    shapes = {}
    shapes['Shapes'] = porous_sample_creator(Rbins=Rbins, 
                                             volume_dict=volume_dict, 
                                             num_of_bubbles=num_of_bubbles, 
                                             x_deform=deforms['x'], 
                                             y_deform=deforms['y'], 
                                             z_deform=deforms['z']).astype(np.uint64)
    porosity = SVMC_sample_porosity(shapes['Shapes'])
    _DUMMY_POSITION_DICT = {'x':int(volume_dict['x']*deforms['x']),
                            'y':int(volume_dict['y']*deforms['y']),
                            'z':int(volume_dict['z']*deforms['z'])}
    SVMC_source(jbase=json_base,
                position=[2, _DUMMY_POSITION_DICT['y']//2, _DUMMY_POSITION_DICT['x']//2 - volume_dict['x']//4],
                direction=[1, 0, 0, -200],
                radius=20)
    jdata.save(shapes,
               'shapes.json',
               opt=dict(compression='zlib', base64=1))
    with open('model.json', 'w') as fp:
        json.dump(json_base, fp, indent=4)
    SVMC_model_and_shapes_merger(str_model = 'model.json',
                                 str_shapes = 'shapes.json',
                                 output_name = 'merged_data_2.json')
    subprocess_list = []
    subprocess_list.append(prog)
    subprocess_list.extend(Modelling_parameters)
    subprocess_list.append(output)
    process = subprocess.run(subprocess_list, 
                             stdout=subprocess.PIPE, 
                             universal_newlines=True,
                             cwd=wd)
    from_jnii = jdata.load(f'{ID}.jnii')
    print('Ending iteration.')
    return from_jnii, porosity

# Дополнительное ###########################################################################################################################
def sphere_creator(Rbins, x_deform=1, y_deform=1, z_deform=1):
    """
    Создает эллипсоид с заданными отношениями полуосей к полному радиусу.
    
    Rbins: int
    Радиус в бинах не считая средний воксель.
    
    x_deform, y_deform, z_deform: float
    Отношение полуосей к полному радиусу.
    
    Возвращает 3D массив с заполненными по правилам SVMC вокселями.
    
    """
    x, y, z = Rbins*x_deform, Rbins*y_deform, Rbins*z_deform
    ellip_base = ellipsoid(x, y, z, levelset=True)
    verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_base, 0)

  
    def Move_to_zero(v):
        return np.array([v.T[0] - np.min(v.T[0]),
                         v.T[1] - np.min(v.T[1]),
                         v.T[2] - np.min(v.T[2])]).T

  
    verts = Move_to_zero(verts)

  
    def centers(v, f):
        cen_and_vox = lambda tri: [np.mean(tri, axis=0), np.floor(np.mean(tri, axis=0)).astype(int)]
        verts_and_faces = v[f]
        return np.array(list(map(cen_and_vox, verts_and_faces)))

  
    c = centers(verts, faces)

  
    def produce_big_number_likewise_no_coords_shift(point, from_to, mean_z, mean_y, mean_x):
        n_not_normalized = np.array([(2*point[2])/(z_deform**2), (2*point[1])/(y_deform**2), (2*point[0])/(x_deform**2)])
        n_normalized = n_not_normalized/np.linalg.norm(n_not_normalized)
        n_normalized_new = np.array([min(math.floor((nn+1)*255/2), 254) for nn in n_normalized])
        shift_cancel = np.array([mean_y, mean_x, mean_z])
        center_new = np.array(point + shift_cancel) - np.floor(np.array(point + shift_cancel))
        center_new = np.floor(center_new * 255)
        whole = np.concatenate([n_normalized_new, [center_new[2], center_new[1], center_new[0]], from_to])
        x = whole
        binary = '0b' + format(int(x[0]),'08b') + format(int(x[1]),'08b') + format(int(x[2]),'08b') + format(int(x[3]),'08b') + format(int(x[4]),'08b') + format(int(x[5]),'08b') + format(int(x[6]),'08b') + format(int(x[7]),'08b')
        while binary[2] != '1':
            binary = binary[:2] + binary[3:]
        return int(binary, 2)


    def centroid_2(v, f):
        ret = []
        c = centers(v, f)
        voxel_number = lambda x: int('0b' + format(int(x[0]),'08b') + format(int(x[1]),'08b') + format(int(x[2]),'08b'), 2)
        vn = np.array(list(map(voxel_number, c[:, 1]))).reshape(-1, 1)
        cvn = np.concatenate((c[:, 0], vn), axis=1)
        cvn = cvn[cvn[:, 3].argsort()]
        cvn_split = np.split(cvn[:, 0:3], np.unique(cvn[:, 3], return_index=True)[1][1:])
        voxel_mean = lambda x: np.mean(x, axis=0)
        vm = np.array(list(map(voxel_mean, cvn_split)))
        cen_and_vox = lambda row: [row, np.floor(row).astype(int)]
        return np.array(list(map(cen_and_vox, vm)))
    coc = centroid_2(verts, faces)


    def Cube_Required(centrs):
        yr = np.max(centrs[:, 1].T[0]).astype(int) + 1
        xr = np.max(centrs[:, 1].T[1]).astype(int) + 1
        zr = np.max(centrs[:, 1].T[2]).astype(int) + 1
        return {"x": xr, "y": yr, "z": zr}
    
  
    xr = Cube_Required(coc)['x']
    yr = Cube_Required(coc)['y']
    zr = Cube_Required(coc)['z']


    def Insider(vol, filler):
        for i in range(vol.shape[0]):
            for j in range(vol.shape[1]):
                if any(vol[i][j] != 1):
                    first = np.argmax(vol[i][j] != 1)
                    last = vol.shape[2] - np.argmax(np.flip(vol[i][j]) != 1) - 1
                    vol[i][j][first+1:last] = [filler for k in range(last - first - 1)]
        return vol
    
  
    def Center_Layer_Filler(vol, cds):
        uwu_y = (np.max(cds[:, 0, 0]) - np.min(cds[:, 0, 0]))/2
        uwu_x = (np.max(cds[:, 0, 1]) - np.min(cds[:, 0, 1]))/2
        uwu_z = (np.max(cds[:, 0, 2]) - np.min(cds[:, 0, 2]))/2
        sp_m = 2
        v_m = 1
        from_to = [v_m, sp_m]
        for whole_point in cds:
            point = np.array([whole_point[0, 0] - uwu_y, 
                              whole_point[0, 1] - uwu_x, 
                              whole_point[0, 2] - uwu_z])
            big_number = produce_big_number_likewise_no_coords_shift(point, from_to, uwu_z, uwu_y, uwu_x)
            vol[whole_point[1, 0].astype(int), 
                whole_point[1, 1].astype(int),
                whole_point[1, 2].astype(int)] = big_number
        Insider(vol, 2)
        for whole_point in cds:
            point = np.array([whole_point[0, 0] - uwu_y, 
                              whole_point[0, 1] - uwu_x, 
                              whole_point[0, 2] - uwu_z])
            big_number = produce_big_number_likewise_no_coords_shift(point, from_to, uwu_z, uwu_y, uwu_x)
            vol[whole_point[1, 0].astype(int), 
                whole_point[1, 1].astype(int),
                whole_point[1, 2].astype(int)] = big_number
        return vol


    volume = np.ones((yr, xr, zr)).astype(np.uint64)
    volume = Center_Layer_Filler(vol = volume, cds = coc)
    
    return volume
  
# Обработка

def relation_calc(whole_area, 
                  deform=1, 
                  what_deforms = 'x', 
                  nphotons=10e10,
                  cube_size=400,
                  det_type='fiber',
                  exclude_center = True,
                  graph_check = False):
    """
    Рассчет коэффициента пропускания/отражения для заданной поверхности детектирования.
    
    whole_area: 2D array
    2D гистограмма распределения фотонов по поверхности.
    
    deform: float
    Отношение полуоси к полному радиусу.
    
    what_deforms: 'x' | 'y' | 'z'
    Какая ось полости деформируется.
    
    nphotons: int
    Число фотонов в моделировании.
    
    cube_size: int
    Размер ребра исследуемого кубического образца.
    
    det_type: 'fiber' | 'line' | 'half' | 'all'
    Тип детектора.
    
    Возвращает коэффициент пропускания/отражения.
    
    """
    scan = whole_area
    xm, ym = np.meshgrid(
        np.arange(0, scan.shape[1], 1), 
        np.arange(0, scan.shape[0], 1)
    )
    if what_deforms == 'x':
        center = [cube_size//2, int(cube_size * deform)//2]
    if what_deforms == 'y':
        center = [int(cube_size * deform)//2, cube_size//2]
    if what_deforms == 'z':
        center = [cube_size//2, cube_size//2]
        
    src_area = (ym - center[0])**2 + (xm - center[1] + cube_size//4)**2 >= 20**2
    
    if det_type == 'fiber':
        det_area = (ym - center[0])**2 + (xm - center[1] - cube_size//4)**2 >= 20**2
    if det_type == 'line':
        det_area = (xm - center[1] - cube_size//4)**2 >= 20**2
    if det_type == 'half':
        det_area = xm <= center[1]
    if det_type == 'all':
        det_area = xm == -1
        
    if exclude_center:
        det_area = det_area | ~src_area
        
    if graph_check:
        plt.figure(figsize=(6, 6))
        sns.heatmap(
                    det_area, 
                    square=True, 
                    cbar=False
                   )
        plt.xticks([])
        plt.yticks([])

    scan_masked = np.ma.masked_array(scan[1:-1, 1:-1], det_area[1:-1, 1:-1])
    all_photons = nphotons
    det_photons = scan_masked.sum()
    rel = det_photons/all_photons
    return rel

def detp_to_hist(detp_data, bins_shape, layer='upp'):
    """
    Рассчитывает гистограмму распределения детектируемых фотонов по заданной поверхности.
    
    detp_data: MCX detp data
    Позиции детектирования фотонов.
    
    bins_shape: [int, int]
    Размеры гистограммы исходя из деформации образца
    
    layer: 'upp' | 'bott'
    Нижний или верхний по отношению к источнику детектор.
    
    Возвращает гистограмму распределения фотонов.
    
    """
    if 'upp':
        mask = detp_data[:, 0] > 1
    if 'bott':
        mask = detp_data[:, 0] < 1
    lay = detp_data[mask]
    layer_hist = np.histogram2d(lay[:, 1], lay[:, 2], bins=bins_shape) 
    return layer_hist[0]

def z_flux_mean(res):
    """
    Сумма потока в каждом срезе вдоль оси z
    
    res: MCX modelling result
    Результат моделирования SVMC.
    
    Возвращает массив значений суммы энергии в срезе.
    
    """
    return res['NIFTIData'].sum(axis=(1, 2))

def sample_sides_hists(detectors_data, graphical_check=False):
    """
    Вычисляет гистограммы распределения по граням образца для моделирования, в котором все стороны - детекторы.
    
    detectors_data: MCX detp data
    Датасет по всем детектируемым фотонам.
    
    graphical_check: bool
    Если True, то будет строить heatmap гистограмм.
    
    Возвращает словарь, в котором для каждой из осей хранится кортеж axis=0 и axis=max гистограмм.
    
    """
    hists_dict = {'z': -1, 'y': -1, 'x': -1}
    positions = detectors_data['MCXData']['PhotonData']['p']
    index_list = [0, 1, 2]
    for index, axis in zip([0, 1, 2], ['z', 'y', 'x']):
        left_border = 0.1
        right_border = np.max(positions[:, index]) - 0.1
        left_mask = positions[:, index] > left_border
        right_mask = positions[:, index] < right_border
        left_layer = positions[left_mask]
        right_layer = positions[right_mask]
        index_list = [0, 1, 2]
        index_list.remove(index)
        left_layer_hist = np.histogram2d(left_layer[:, index_list[0]], 
                                         left_layer[:, index_list[1]], 
                                         bins=[np.max(left_layer[:, index_list[0]]).astype(int), 
                                               np.max(left_layer[:, index_list[1]]).astype(int)])[0]
        right_layer_hist = np.histogram2d(right_layer[:, index_list[0]], 
                                          right_layer[:, index_list[1]], 
                                          bins=[np.max(right_layer[:, index_list[0]]).astype(int), 
                                                np.max(right_layer[:, index_list[1]]).astype(int)])[0]
        hists_dict[axis] = (left_layer_hist, right_layer_hist)
        if graphical_check:
            plt.figure(figsize=(10, 5))
            plt.suptitle(axis, fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title("Left")
            sns.heatmap(left_layer_hist, square=True, norm=colors.LogNorm())
            plt.tight_layout()
            plt.subplot(1, 2, 2)
            plt.title("Right")
            sns.heatmap(right_layer_hist, square=True, norm=colors.LogNorm())
            plt.tight_layout()
    return hists_dict
