from get_url_links import get_saved_urls
import networkx as nx
from lxml import html
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


raw_text = '<html><head></head><body><a>FooBar</a></body></html>'


def dfs(node, graph, tag):
  tag[node] = node.tag
  for child in node.getchildren():
    graph.add_edge(node, child)
    dfs(child, graph, tag)


graph = nx.DiGraph()
tag = {}
html_tag = html.document_fromstring(raw_text)
dfs(html_tag, graph, tag)

pos = graphviz_layout(graph, prog='dot')

label_props = {'size': 16,
               'color': 'black',
               'weight': 'bold',
               'horizontalalignment': 'center',
               'verticalalignment': 'center',
               'clip_on': True}
bbox_props = {'boxstyle': "round, pad=0.2",
              'fc': "grey",
              'ec': "b",
              'lw': 1.5}

nx.draw_networkx_edges(graph, pos, arrows=True)
ax = plt.gca()

for node, label in tag.items():
        x, y = pos[node]
        ax.text(x, y, label,
                bbox=bbox_props,
                **label_props)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

plt.savefig('sample_DOM.pdf')
plt.show()

exit(0)

urls = get_saved_urls() 
for url in urls: 
  if url is not None:
    try:
      req = requests.get(url, timeout=0.5, stream=True)
      soup = BeautifulSoup(req.text)

      break
    except:
      pass

print(urls[:10])
