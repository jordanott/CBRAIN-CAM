data_opts = {
	'8col':{
		'train':{
			'feature_fn': 'full_physics_essentials_train_month01_shuffle_features.nc',
			'target_fn': 'full_physics_essentials_train_month01_shuffle_targets.nc'
		},
		'test':{
			'feature_fn': 'full_physics_essentials_valid_month02_features.nc',
			'target_fn': 'full_physics_essentials_valid_month02_targets.nc'
		},
		'norm_fn': 'full_physics_essentials_train_month01_norm.nc',
        'input_shape':94,
        'output_shape':65,
	},
	'32col':{
		'train':{
			'feature_fn': 'full_physics_essentials_train_month01-06_shuffle_features.nc',
			'target_fn': 'full_physics_essentials_train_month01-06_shuffle_targets.nc'
		},
		'test':{
			'feature_fn': 'full_physics_essentials_valid_month07-12_features.nc',
			'target_fn': 'full_physics_essentials_valid_month07-12_targets.nc'
		},
		'norm_fn': 'full_physics_essentials_train_month01-06_norm.nc',
        'input_shape':64,
        'output_shape':65,
	},
    'land_data':{
        'train':{
            'feature_fn':'full_physics_essentials_train_month01_shuffle_features.nc',
            'target_fn':'full_physics_essentials_train_month01_shuffle_targets.nc'
        },
        'test':{
            'feature_fn':'full_physics_essentials_valid_month02_features.nc',
            'target_fn':'full_physics_essentials_valid_month02_targets.nc'
        },
        'norm_fn':'full_physics_essentials_train_month01_norm.nc',
        'input_shape':64,
        'output_shape':65,
    },
    'fluxbypass_aqua':{
        'input_shape':304,
        'output_shape':218,
    }


}
