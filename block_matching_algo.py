p1 = [438,17,776,30,800,359,450,357]
p2 = [450,10,810,360]

p1a = [min(p1[0::2]),min(p1[1::2]),max(p1[0::2]),max(p1[1::2])]
print(p1a)

p_overlap = [max(p1a[0], p2[0]), max(p1a[1], p2[1]), min(p1a[2], p2[2]), min(p1a[3], p2[3])]

area_overlap = (p_overlap[3]-p_overlap[1]) * (p_overlap[2]-p_overlap[0])
area_total = (p1a[3]-p1a[1]) * (p1a[2]-p1a[0])

print(area_total, area_overlap, area_overlap/area_total*100)