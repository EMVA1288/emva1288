import emva1288
from emva1288 import process
from emva1288 import report
import os


def main(dataset_descripton_file,
         gain,
         black_level,
         bit_depth,
         steps
         ):
    # Load the dataset
    parser = process.ParseEmvaDescriptorFile(dataset_descripton_file)
    imgs = process.LoadImageData(parser.images)
    dat = process.Data1288(imgs.data)

    # Description of the setup
    setup = report.info_setup()
    setup['Standard version'] = emva1288.__standard_version__

    # Basic information
    basic = report.info_basic()
    basic['vendor'] = 'Simulation'
    basic['data_type'] = 'Single'
    basic['sensor_type'] = 'simulated sensor'
    basic['resolution'] = f'{dat.cols}x{dat.rows}'
    basic['model'] = 'Simulated camera'

    # Marketing information
    marketing = report.info_marketing()
    marketing['watermark'] = 'Example'

    # Initialize the report with the marketing data
    # Provide a non existent name for the output directory
    myreport = report.Report1288('myreport',
                                 marketing=marketing,
                                 setup=setup,
                                 basic=basic)

    # Operation point
    # bit_depth, gain, black_level, exposure_time, wavelength,
    # temperature, housing_temperature, fpn_correction,
    # summary_only

    op1 = report.info_op()
    op1['summary_only'] = False
    op1['camera_settings']['Gain'] = gain
    op1['camera_settings']['Black level'] = black_level
    op1['camera_settings']['Bit depth'] = f'{bit_depth} bits'
    op1['test_parameters']['Illumination'] = 'Variable with constant exposure time'
    op1['test_parameters']['Irradiation steps'] = steps

    # Add the operation point to the report
    # we can add as many operation points as we want
    # we pass the emva1288.Data1288 object to extract automatically all the results
    # and graphics
    myreport.add(op1, dat.data)

    # Generate the latex files
    myreport.latex()


if __name__ == '__main__':
    # Arguments used for dataset generation and report configuration
    gain = 0.1
    black_level = 29.4
    bit_depth = 12
    steps = 50

    # # Alternative 1: Load one test to add it as operation point
    # dir_ = '/path/to/emva1288/datasets/'
    # fname = 'EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt'
    # fname = os.path.join(dir_, fname)

    # Alternative 2: Generate a new test dataset
    from emva1288.camera.dataset_generator import DatasetGenerator

    dataset_generator = DatasetGenerator(width=640,
                                         height=480,
                                         K=gain,
                                         blackoffset=black_level,
                                         bit_depth=bit_depth,
                                         steps=steps,
                                         exposure_fixed=1000000,
                                         dark_current_ref=30)
    fname = dataset_generator.descriptor_path

    main(dataset_descripton_file=fname,
         gain=gain,
         black_level=black_level,
         bit_depth=bit_depth,
         steps=steps)
