import nose


default_test_modules = ['emva1288.tests.test_coding_standards', ]


def run():
    nose.main(defaultTest=default_test_modules)

if __name__ == '__main__':
    run()