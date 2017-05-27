vocabulary = containers.Map; 
fid=fopen('vocabulary.txt','r+','n','utf-8');
tline=fgetl(fid);
tline=native2unicode(tline);
s = regexp(tline, ',', 'split');
k = s{1};
v = s{2};
vocabulary(k) = v;
idx = 0;
while tline
    idx = idx + 1
    
    tline=fgetl(fid);
    tline=native2unicode(tline);
    s = regexp(tline, ',', 'split');
    k = s{1};
    v = s{2};
    vocabulary(k) = v;
end
fclose(fid);
