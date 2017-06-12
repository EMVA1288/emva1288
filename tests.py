import nose
import os


default_test_modules = ['emva1288.unittests.test_coding_standards',
                        'emva1288.unittests.test_camera',
                        'emva1288.unittests.test_parser.TestParser',
                        'emva1288.unittests.test_loader.TestLoader',
                        'emva1288.unittests.test_data.TestData',
                        'emva1288.unittests.test_results.TestResults',
                        'emva1288.unittests.test_report.TestReportGenerator']


def run():
    os.environ["NOSE_WITH_COVERAGE"] = "1"
    os.environ['NOSE_COVER_PACKAGE'] = 'emva1288'
    os.environ["NOSE_COVER_HTML"] = "1"
    os.environ["NOSE_COVER_ERASE"] = "1"
    nose.main(defaultTest=default_test_modules)


if __name__ == '__main__':
    run()
