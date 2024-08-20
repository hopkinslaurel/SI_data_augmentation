# defines file paths for the given task

def get_paths(task):
	paths = {
		'treecover': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/data/png/contus_uar/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/treecover/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/treecover/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_contus_uar_Apr-06-2022.txt',
		},

		'nightlights': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/data/png/contus_pop/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/nightlights/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/nightlights/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_contus_pop_Nov-16-2022.txt',
		},

		'elevation': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/data/png/contus_uar/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/elevation/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/elevation/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_contus_uar_Apr-06-2022.txt',
		},

		'population': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/data/png/contus_pop/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/population/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/mosaiks/population/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_contus_pop_Nov-16-2022.txt',
		},

		'landuse': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/ucMerced_landuse/data/npy/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/ucMerced_landuse/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/ucMerced_landuse/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_ucMerced_landuse_Jan-05-2023.txt',
		},

		'residential': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/ucMerced_landuse/data/npy/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/ucMerced_residential/runs/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/ucMerced_residential/models/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_ucMerced_residential_Jan-10-2023.txt',
		},

		'coffee': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/coffee/data/jpg/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/coffee/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/coffee/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_coffee_Aug-18-2024.txt',
		},

		'eurosat': {
			'home_dir': '/nfs/stak/users/hopkilau/shallow/',
			'img_dir': '/nfs/hpc/share/hopkilau/datadrive/eurosat/data/eurosat_ms/npy/',
			'tif_dir': '/nfs/hpc/share/hopkilau/datadrive/eurosat/data/eurosat_ms/tif/',
			'npy_dir': '/nfs/hpc/share/hopkilau/datadrive/eurosat/data/eurosat_ms/npy/',
			'log_dir': '/nfs/hpc/share/hopkilau/datadrive/eurosat/runs/cutmix/',
			'model_dir': '/nfs/hpc/share/hopkilau/datadrive/eurosat/models/cutmix/',
			'means': '/nfs/stak/users/hopkilau/shallow/channel_means/channel_means_eurosat_ms_Jul-01-2024.txt',
		}
	}

	return paths[task]

