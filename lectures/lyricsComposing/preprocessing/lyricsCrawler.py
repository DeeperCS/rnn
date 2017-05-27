import urllib.request
from lxml import etree

url = 'http://music.baidu.com/search/lrc?key=%E5%91%A8%E6%9D%B0%E4%BC%A6'
response = urllib.request.urlopen(url)
resp_text = response.read().decode('UTF-8')

html = etree.HTML(resp_text)

lyricList = []
p_list = html.xpath('//div[@class="lrc-content"]//p')
for p in p_list:
    lyric = [s for s in p.itertext()]
    lyricList.append(lyric)
    print(lyric)
    
# Write to file
for idx, lyric in enumerate(lyricList):
    filename = "lyrics/lyric-{}.txt".format(idx)
    with open(filename, 'wb') as f:
        for line in lyric:
            # print(line)
            f.write(line.encode("utf-8"))    
    # break
print('finished')