import pep8
import os
import emva1288
from nose.tools import assert_equal

PEP8_ADDITIONAL_IGNORE = []
EXCLUDE_FILES = []


def test_pep8_conformance():

    dirs = []
    dirname = os.path.dirname(emva1288.__file__)
    dirs.append(dirname)
    examplesdir = os.path.join(dirname, '..', 'examples')
    examplesdir = os.path.abspath(examplesdir)
    dirs.append(examplesdir)

    pep8style = pep8.StyleGuide()

    # Extend the number of PEP8 guidelines which are not checked.
    pep8style.options.ignore = (pep8style.options.ignore +
                                tuple(PEP8_ADDITIONAL_IGNORE))
    pep8style.options.exclude.extend(EXCLUDE_FILES)

    result = pep8style.check_files(dirs)
    msg = "Found code syntax errors (and warnings)."
    assert_equal(result.total_errors, 0, msg)

if __name__ == '__main__':
    import nose
    nose.runmodule()
