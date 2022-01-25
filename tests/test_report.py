import pytest
import tempfile
import os
from emva1288.report import info_op, Report1288
from emva1288.process.plotting import EVMA1288plots


def test_report_generation(data):
    dataset, parser, loader, data = data
    op = info_op()
    with tempfile.TemporaryDirectory() as outdir:
        # create report
        report = Report1288(outdir)
        report.add(op, data.data)
        # check that output directory has been created
        for directory in ('files', 'OP1', 'upload'):
            assert os.path.isdir(os.path.join(outdir, directory))
        # check that plots have been created and saved
        for plt in EVMA1288plots:
            name = plt.__name__
            path = os.path.join(outdir, 'OP1', name + '.pdf')
            assert os.path.isfile(path)
        # generate tex files
        report.latex()
        # check that tex files have been created
        for fil in ('emvadatasheet.sty', 'report.tex'):
            path = os.path.join(outdir, fil)
            assert os.path.isfile(path)
