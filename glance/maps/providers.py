
ARCGIS = {
    'name': 'arcgis',
    'styles': {"physical": 'World_Physical_Map', "terrain": 'World_Terrain_Base', "topo": 'World_Topo_Map'},
    'url_format': "http://{{s}}.arcgisonline.com/ArcGIS/rest/services/{style}/MapServer/tile/{{z}}/{{y}}/{{x}}.{extension}",
    'extension': 'jpg',
    'sub_domains': ['services'], #['server']
}

CARTODB = {
    'name': 'cartodb',
    'styles': {"dark_all": 'dark_all', "light_all": 'light_all'},
    'url_format': "http://{{s}}.basemaps.cartocdn.com/{style}/{{z}}/{{x}}/{{y}}.png",
    'extension': 'png',
    'sub_domains': ['a','b','c','d',],
}
