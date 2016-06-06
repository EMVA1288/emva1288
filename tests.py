import nose


default_test_modules = ['emva1288.unittests.test_coding_standards',
                        'emva1288.unittests.test_camera']


def run():
    nose.main(defaultTest=default_test_modules)

if __name__ == '__main__':
    run()
