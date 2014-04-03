# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can be
# found in the LICENSE file.

"""Template function to generate repport

"""
#
#
# def generate_latex(results, outputdir=-1, extra_plots=False):
#     '''Write the tex files and generate the plots
#     needed for the template compilation
#
#     results: Results1288 object
#     outputdir:
#         * -1 Just print the text files
#         * path write the tex files in the given path
#     extra_plots:
#         * True: generate all the plots and add them to the generated tex files
#         * False: print the content of the tex files
#     '''
#
#     if extra_plots:
#         results.data.data['datasheet']['operation_point']['ExtraPlots'] = 'True'
#     latex = results.latex()
#     if outputdir == -1:
#         for k, v in latex.items():
#             if not v:
#                 continue
#             print '\n\n', '-' * 50
#             print '-'
#             print '- FILE:', k + '_data'
#             print '-'
#             print '-' * 20
#             print v
#             print '-' * 50
#
#     elif os.path.isdir(outputdir):
#         # TODO: Clean up the operation point latex data management
#         fname = None
#         #check if there is an operation_point_datax.tex file available
#         for i in range(1, 11):
#             tname = os.path.join(outputdir,
#                                  'operation_point_data%d.tex' % i)
#             if not os.path.isfile(tname):
#                 fname = tname
#                 break
#
#         if not fname:
#             print 'Operation point files 1-10 are used'
#             return
#
#         file_ = open(fname, 'w')
#         file_.write(latex['operation_point'])
#
#         img_d = os.path.join(outputdir, 'Images')
#         if os.path.isdir(img_d):
#             plots = plotting.Plotting1288V3(results)
#             if extra_plots:
#                 plots.plot()
#             else:
#                 plots.plot(2, 3)
#
#             t = '\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
#             t += '% Graphics from Python\n\n'
#             for plot_name, figure in plots.figures.items():
#                 varname = plot_name.replace('_', '')
#                 figure.canvas.print_eps(os.path.join(img_d, plot_name + '.eps'))
#                 t += '\\def\\%sPlot{%s.eps}\n' % (varname, plot_name)
#
#             file_.write(t)
#
#         else:
#             print 'No Images directory to save snr and ptc graphics'
#         file_.close()
#
#         if latex['camera']:
#             fname = os.path.join(outputdir, 'camera_data.tex')
#             file_ = open(fname, 'w')
#             file_.write(latex['camera'])
#             file_.close()
#     else:
#         print 'Invalid directory to create tex file'
