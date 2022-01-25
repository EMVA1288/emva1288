from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils import nodes
from collections import OrderedDict


def setup(app):
    app.add_directive('emva1288', Emva1288Directive)
    return {'version': '0.1'}


class Emva1288Directive(Directive):
    """Sphinx directive for EMVA1288 results

    Process emva blocks to extract information relative to the result, this
    information includes

    - section: Section of the emva standard that includes this value
    - short: Short description of the value
    - symbol: Latex symbol for the value
    - unit: Unit of the value
    - latexname: Name to be used when refering the value inside the latex reports

    Example
    -------

    ::

        .. emva1288::
            ::Section: spatial
            :Short: PRNU
            :Symbol: $PRNU_{1288}$
            :Unit: \%
            :LatexName: PRNU

    """
    has_content = True
    option_spec = OrderedDict()
    option_spec['section'] = directives.unchanged_required
    option_spec['short'] = directives.unchanged_required
    option_spec['symbol'] = directives.unchanged
    option_spec['unit'] = directives.unchanged
    option_spec['latexname'] = directives.unchanged

    def run(self):
        idb = nodes.make_id("emva1288-" + self.options['section'])
        section = nodes.section(ids=[idb])
        section += nodes.rubric(text='Emva1288')
        lst = nodes.bullet_list()

        for k in self.option_spec.keys():
            if k not in self.options:
                continue

            item = nodes.list_item()
            item += nodes.strong(text=k + ':')
            item += nodes.inline(text=' ' + self.options[k])
            lst += item
        section += lst
        return [section]
