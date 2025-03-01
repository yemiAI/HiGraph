
##classifier??
annotation_dictionary = {
    ##follower
    #1
    "jesse_SR_SL_000_F.bvh": [
        ["SR", [55, 89]],
        ["SL", [89, 120]],
        ["SR", [120, 157]],##fix
        ["SL", [157, 187]],
        ["SR", [187, 221]],
        ["SL", [221, 250]],

    ],
    #2
    "jesse_SL_SR_000_F.bvh": [
        ["SL", [60, 87]],
        ["SR", [87, 120]],
        ["SL", [120, 151]],
        ["SR", [151, 185]],
        ["SR", [185, 218]],
        ["SR", [218, 250]]

    ],
    #3
    "jesse_SL_CR_SL_CR_CL_SL_000_F.bvh": [
        ["SL", [25, 69]],
        ["CR", [69, 89]],
        ["SL", [89, 140]],
        ["CR", [140, 164]],
        ["CL", [164, 185]],
        ["SL", [185, 230]]
    ],
    #4
    "jesse_SR_CL_SR_CL_CR_SL_000_F.bvh": [
        ["SR", [0, 56]],
        ["CL", [56, 78]],
        ["SR", [78, 126]],
        ["CL", [126, 150]],
        ["CR", [150, 189]],
        ["SL", [189, 232]],
        # ["SR", [189, 233]]
    ],
    #5
    "jesse_SL_SR_CL_SR_SL_CR_SL_000_F.bvh": [
        ["SL", [0, 48]],
        ["SR", [48, 83]],
        ["CL", [83, 101]],
        ["SR", [101, 142]],
        ["SL", [142, 189]],
        ["CR", [189, 210]],
        ["SL", [210, 244]],
    ],
    #6
    "jesse_BL_BR_FL_FR_BL_000_F.bvh": [
        ["BL", [36, 98]],
        ["BR", [98, 128]],
        ["FL", [128, 177]],
        ["FR", [177, 208]],
        ["BL", [208, 236]],
    ],
    #7
    "jesse_BR_BL_FR_FL_BR_BL_FR_000_F.bvh": [
        ["BR", [8, 47]],
        ["BL", [47, 68]],
        ["FR", [68, 112]],
        ["FL", [112, 138]],
        ["BR", [138, 178]],
        ["BL", [178, 202]],
        ["FR", [202, 246]]
    ],
    #8
    "jesse_BL_FR_SL_CR_SL_FR_SL_000_F.bvh": [
        ["BL", [15, 50]],
        ["FR", [50, 89]],
        ["SL", [89, 129]],  ## recapture poor arm tracks
        ["CR", [129, 153]],
        ["SL", [153, 188]],
        ["FR", [188, 223]],
        ["SL", [223, 256]],
    ],
    #9 bad data --delete this at some point
    # "jesse_BL_FR_SL_CR_000.bvh": [
    #     ["BL", [0, 0]],
    #     ["FR", [0, 0]],
    #     ["SL", [0, 0]],
    #     ["CR", [0, 0]],
    # ],
    # #10
    "jesse_BL_BR_SL_FR_FL_SR_SL_000_F.bvh": [
        ["BL", [0, 48]],
        ["BR", [48, 77]],
        ["SL", [77, 117]],
        ["FR", [117, 149]],
        ["FL", [149, 181]],
        ["SR", [181, 212]],
        ["SL", [212, 249]]
    ],
    #11
    "jesse_BR_BL_SR_FL_FR_000_F.bvh": [
        ["BR", [57, 100]],
        ["BL", [100, 126]],
        ["SR", [126, 162]],
        ["FL", [162, 202]],
        ["FR", [202, 234]],
    ],
    #12
    "jesse_BL_BR_BL_FR_SL_000_F.bvh": [
        ["BL", [48, 81]],
        ["BR", [81, 116]],
        ["BL", [116, 142]],
        ["FR", [142, 180]],
        ["FL", [180, 215]],
    ],
    #13
    "jesse_BR_BL_BR_FL_FR_000_F.bvh": [
        ["BR", [13, 59]],
        ["BL", [59, 93]],
        ["BR", [93, 124]],
        ["FL", [124, 166]],
        ["FR", [166, 223]],
    ],
    #14
    "jesse_CL_BR_BL_CR_FL_FR_000_F.bvh": [
        ["CL", [34, 66]],
        ["BR", [66, 109]],
        ["BL", [109, 142]],
        ["CR", [142, 165]],
        ["FL", [165, 213]],
        ["FR", [213, 246]]
    ],
    #15
    "jesse_CR_BL_BR_CL_FR_FL_000_F.bvh": [
        ["CR", [0, 30]],
        ["BL", [30, 73]],
        ["BR", [73, 116]],
        ["CL", [116, 133]],
        ["FR", [133, 177]],
        ["FL", [177, 218]]
    ],

    # jesse_leader
    # 1
    "jesse_SL_SR_SL_SR_SL_SR_000_L.bvh": [
        ["SL", [26, 64]],
        ["SR", [64, 95]],
        ["SL", [95, 127]],
        ["SR", [127, 163]],
        ["SL", [163, 193]],
        ["SR", [193, 229]],
    ],
    # 2
    "jesse_SR_SL_SR_SL_SR_SL_000_L.bvh": [
        ["SR", [37, 87]],
        ["SL", [87, 120]],
        ["SR", [120, 154]],
        ["SL", [154, 187]],
        ["SR", [187, 221]],
        ["SL", [221, 250]],
    ],
    # 3
    "jesse_SL_CR_SL_CR_CL_SL_000_L.bvh": [
        ["SL", [0, 54]],
        ["CR", [54, 72]],
        ["SL", [72, 120]],
        ["CR", [120, 159]],
        ["CL", [159, 168]],
        ["SL", [168, 211]],
    ],
    # 4
    "jesse_SR_CL_SR_CL_CR_SL_000_L.bvh": [
        ["SR", [0, 55]],
        ["CL", [55, 62]],
        ["SR", [62, 98]],
        ["CL", [98, 109]],
        ["CR", [109, 122]],
        ["SL", [122, 172]],
        ["CR", [172, 184]],
        ["SL", [184, 233]]
    ],
    # 5
    "jesse_SL_SR_CL_SR_SL_CR_SL_000_L.bvh": [
        ["SL", [49, 100]],
        ["SR", [100, 152]],
        ["CL", [152, 165]],
        ["SR", [165, 205]],
        ["SL", [205, 245]],
    ],
    # 6 BAD DATA But try to make sense out of it
    "jesse_FR_FL_BR_BL_FR_000_L.bvh": [
        ["FR", [29, 55]],
        ["FL", [92, 122]],
        ["BR", [122, 158]], ###bad data
        ["BL", [158, 186]],
        ["FR", [186, 228]],
        #["SL", []]
    ],
    # 7
    "jesse_FL_FR_BL_BR_FL_FR_000_L.bvh": [
        ["FL", [21, 65]],
        ["FR", [65, 92]],
        ["BL", [92, 125]],
        ["BR", [125, 159]], ###bad data
        ["FL", [159, 195]],
        ["FR", [195, 223]],
    ],
    # 8
    "jesse_FL_BR_SL_CR_FL_000_L.bvh": [
        ["FL", [51, 98]],
        ["BR", [98, 129]],
        ["SL", [129, 175]],
        ["CR", [175, 183]],
        ["FL", [183, 226]],
    ],
    # 9
    "jesse_FR_BL_SR_CL_FR_BL_000_L.bvh": [
        ["FR", [33, 66]],
        ["BL", [66, 104]],
        ["SR", [104, 134]],
        ["CL", [134, 152]],
        ["FR", [152, 188]],
        ["BL", [188, 225]],
    ],
    # 10
    "jesse_FL_FR_SL_BR_BL_SR_FL_000_L.bvh": [

        ["FL", [5, 46]],
        ["FR", [46, 75]],
        ["SL", [75, 108]],
        ["BR", [108, 141]],
        ["BL", [141, 173]],
        ["SR", [173, 211]],
        ["FL", [211, 243]]
    ],
    # 11
    "jesse_FR_FL_SR_BL_BR_SL_FR_000_L.bvh": [
        ["FR", [10, 42]],
        ["FL", [42, 67]],
        ["SR", [67, 99]],
        ["BL", [99, 135]],
        ["BR", [135, 165]],
        ["SL", [173, 205]],
        ["FR", [205, 239]],
    ],
    # 12
    "jesse_FL_FR_FL_BR_BL_BR_BL_000_L.bvh": [
        ["FL", [12, 55]],
        ["FR", [55, 88]],
        ["FL", [88, 116]],
        ["BR", [116, 149]],
        ["BL", [149, 183]],
        ["BR", [183, 215]],
        ["BL", [215, 253]]
    ],
    # 13
    "jesse_FR_FL_FR_BL_BR_BL_000_L.bvh": [
        ["FR", [34, 65]],
        ["FL", [65, 95]],
        ["FR", [95, 124]],
        ["BL", [124, 159]],
        ["BR", [159, 191]],
        ["BL", [191, 228]],
    ],
    # # 14
    # "": [
    #
    # ],
    # 15
    "jesse_CR_FL_FR_CL_BR_BL_000_L.bvh": [
        ["CR", [6, 37]],
        ["FL", [37, 79]],
        ["FR", [79, 118]],
        ["CL", [118, 133]],
        ["BR", [133, 179]],
        ["BL", [179, 215]]
    ],
    # 16
    "jesse_CL_FR_FL_CR_BL_BR_000_L.bvh": [
        ["CL", [30, 58]],
        ["FR", [58, 89]],
        ["FL", [89, 117]],
        ["CR", [117, 155]],
        ["BL", [155, 193]],
    ],


}


defaulted_annotation_dictionary = {}

def inserted_entry(start_frame, entry, default_anno = 'NA'):
    if (entry[1][0] > start_frame):
        new_entry = [default_anno, [start_frame, entry[1][0]]]
        return new_entry
    else:
        return None


for k in annotation_dictionary:
    origlist = annotation_dictionary[k]
    defaulted_annotation_dictionary[k] = []

    start_frame = 0
    for entry in origlist:
        new_entries = inserted_entry(start_frame, entry)
        if (new_entries is not None):
            defaulted_annotation_dictionary[k].append(new_entries)
        defaulted_annotation_dictionary[k].append(entry)
        start_frame = entry[1][1]

#print(defaulted_annotation_dictionary)
