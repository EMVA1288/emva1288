import sample_report

from emva1288.camera.dataset_generator import DatasetGenerator

gain = 0.1
black_level = 29.4
bit_depth = 12
steps = 10

dataset_generator = DatasetGenerator(width=32,
                                     height=24,
                                     K=gain,
                                     blackoffset=black_level,
                                     bit_depth=bit_depth,
                                     steps=steps,
                                     exposure_fixed=1000000,
                                     dark_current_ref=30)

sample_report.main(dataset_descripton_file=dataset_generator.descriptor_path,
                   gain=gain,
                   black_level=black_level,
                   bit_depth=bit_depth,
                   steps=steps)
