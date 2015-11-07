import csv
import xml.etree.ElementTree as ET

def get_scaling_factor(media_num):
    return media_ref_of_interest[media_num]["msEnd"] - media_ref_of_interest[media_num]["msStart"] + 1

def convert_to_frames(value, scaling_factor, num_frames):
    return int(round(float(value) / scaling_factor * (num_frames - 1)) + 1)

def convert_to_ms(value, offset, scaling_factor, num_frames):
    return int(round(float(value - 1 + offset) / (num_frames - 1) * scaling_factor))
    
def print_coding_scheme(coding_scheme):
    for datatype in coding_scheme:
        print datatype + ", " + coding_scheme[datatype]["name"]
        for value in coding_scheme[datatype]:
            if value != "name":
                print "    " + value, coding_scheme[datatype][value]

# coding_scheme format:
# field id1 -> {name: 'label for field 1', 1: descriptionOfMeaningOfOneInField1, 2: descriptionOfMeaningOfTwoInField1}
# field id2 -> {name: 'label for field 2', 1: descriptionOfMeaningOfOneInField2, 2: descriptionOfMeaningOfTwoInField2}
def populate_coding_scheme(root):
    coding_scheme = {} 
    for record in root:
        if (record.tag == "CODING-SCHEME"):
            for field in record:
                assert(field.attrib.get('ID') not in coding_scheme)
                coding_scheme[field.attrib.get('ID')] = {}
                coding_scheme[field.attrib.get('ID')]["name"] = field.attrib.get('NAME')
                for value in field:
                    coding_scheme[field.attrib.get('ID')][value.attrib.get('ID')] = value.attrib.get('NAME')
    return coding_scheme

def is_gloss_but_not_english_translation(machine_annotation_type):
    return int(machine_annotation_type) >= 10000 and int(machine_annotation_type) != 20001
    
def parse_element(elem, machine_annotation_type):
    if ("VID" in elem.attrib):
        assert(not elem.text)
        label = "(" + coding_scheme[machine_annotation_type][elem.attrib.get("VID")] + ")"
        start = (int)(elem.attrib.get("S"))
        end = (int)(elem.attrib.get("E"))
    else:
        label = elem.text
        start = (int)(elem.attrib.get("S"))
        end = (int)(elem.attrib.get("E"))
    return (label, start, end)

def is_english_translation(machine_datatype):
    return machine_datatype == "20001"

# Maintain an ongoing gloss for each millisecond, separated
# by utter_type (e.g., 'main gloss' or 'literal translation'),
# inside of the utter_rep representation of the utterance.
# Does so by iterating over all the milliseconds between start
# and end and appending the new parameter label to the existing label.
# Unlike the non-gloss items, we perform the expansion into each
# millisecond early for the gloss items so that we can maintain
# proper string-appending state (it's less easy to push the 
# expansion process to file writing time for the strings).
def append_gloss_to_all_intervening_milliseconds(utter_rep, utter_type, label, start, end):
    for msLabel in range(start, end+1):
        if utter_type not in utter_rep:
            utter_rep[utter_type] = {}
        if msLabel not in utter_rep[utter_type]:
            utter_rep[utter_type][msLabel] = ""  
        existingLabel = utter_rep[utter_type][msLabel]  
        utter_rep[utter_type][msLabel] = " ".join([existingLabel, label])
    return utter_rep

# Small form representation of an utterance:
#   "type -> (value)": {start: #, end: #} 
# It is "small" because we don't expand all the intervening milliseconds
# between start and end
def store_small_form(utter_rep, human_annotation_type, label, start, end):
    utter_rep[human_annotation_type + " => " + label] = {}
    utter_rep[human_annotation_type + " => " + label]["start"] = start
    utter_rep[human_annotation_type + " => " + label]["end"] = end
    return utter_rep

# At the global level, populate the meta knowledge about this
# track with its (a) English translation, (b) start ms, (c) end ms.
# This helps with converting to frames.
def populate_meta_knowledge(label, start, end):
    # Make sure we don't overwrite any existing data
    assert ("msStart" not in media_ref_of_interest[media_num])
    assert ("msEnd" not in media_ref_of_interest[media_num])
    assert ("english" not in media_ref_of_interest[media_num])
    # Update the global info for this track
    media_ref_of_interest[media_num]["msStart"] = start
    media_ref_of_interest[media_num]["msEnd"] = end
    media_ref_of_interest[media_num]["english"] = label

# Using the coding scheme, parse the XML and store in our internal
# format, in which glosses have been expanded to each intervening
# millisecond and which glosses occur at that time, and the 
# non-glosses have a "type -> (value)": {start: #, end: #} format. 
# Example output utterance representation:
# {
#     'eye brows => (lowered lid)': {'start': 34, 'end': 234}, 
#     'main gloss': {300: ' fs-JOHN', 301: ' fs-JOHN', 302: ' fs-JOHN', 303: ' fs-JOHN',    ..... }
# }
def populate_utterance_representation(utterance, coding_scheme):
    utter_rep = {}
    for track in utterance.findall("SEGMENT/TRACK"):
        machine_annotation_type = track.attrib.get("FID")
        human_annotation_type = coding_scheme[machine_annotation_type]["name"]
        for elem in track:
            (annotation_value, start, end) = parse_element(elem, machine_annotation_type)

            if is_gloss_but_not_english_translation(machine_annotation_type):
                utter_rep = append_gloss_to_all_intervening_milliseconds(utter_rep, human_annotation_type, annotation_value, start, end)
            else:
                utter_rep = store_small_form(utter_rep, human_annotation_type, annotation_value, start, end)

            if is_english_translation(machine_annotation_type):
                populate_meta_knowledge(annotation_value, start, end)
                
    return utter_rep

def write_outputs(path, fn, utter_rep):
    write_truth_csv(path + fn + "_truth.csv", utter_rep)
    append_to_overview(path + "overview.csv")

def write_truth_csv(filename, utter_rep):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for token in utter_rep:
            scaling = get_scaling_factor(media_num)
            nframes = media_ref_of_interest[media_num]["frames"]
            if "=>" in token:
                start = utter_rep[token]["start"]
                end = utter_rep[token]["end"]

                startFrame = convert_to_frames(start, scaling, nframes)
                endFrame = convert_to_frames(end, scaling, nframes)

                #print u"%6d-%6d %s" % (startFrame, endFrame, token)

                writer.writerow([token] + \
                                    ['0'] * (startFrame - 1) + \
                                    ['1'] * (endFrame - startFrame + 1) + \
                                    ['0'] * (nframes - endFrame))
            else:
                resultStr = []
                resultStr += [token]
                for i in range(1, nframes + 1):
                    frameIdx = convert_to_ms(i, 0.5, scaling, nframes)
                    if frameIdx in utter_rep[token]:
                        resultStr += [utter_rep[token][frameIdx]]
                    else:
                        resultStr += ['']
                    #TODO currently just grabs on the frames; can improve this to get the union 
                    # of all tokens in the gloss between the ms that corresponds to start/end of frame
                    # For instance, BOOK in ncslgr10a-002-fs-JOHN_FINISH_READ_2_BOOK_ is listed
                    # as only one frame, when really it should be 2 or 3
                writer.writerow(resultStr)

def append_to_overview(overview_filename):
    with open(overview_filename, "ab") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([fn + "_truth.csv"] + [media_ref_of_interest[media_num]["english"]] + 
                        [media_ref_of_interest[media_num]["frames"]] + [media_ref_of_interest[media_num]["msStart"]] +
                        [media_ref_of_interest[media_num]["msEnd"]] + [media_ref_of_interest[media_num]["fnOurs"]] +
                        [media_num])

        

media_ref_of_interest = {}
media_ref_of_interest["485"] = {}
media_ref_of_interest["485"]["fnOurs"] = "ncslgr10a-002-fs-JOHN_FINISH_READ_2_BOOK_"
media_ref_of_interest["485"]["frames"] = 76
#media_ref_of_interest["497"] = {}
#media_ref_of_interest["497"]["fnOurs"] = "ncslgr10a-005-fs-JOHN_FUTURE_FINISH_READ_2_"
#media_ref_of_interest["497"]["frames"] = 94
#media_ref_of_interest["520"] = {}
#media_ref_of_interest["520"]["fnOurs"] = "ncslgr10a-013-fs-JOHN_READ_2_BOOK_nd-flat-OABOUT_"
#media_ref_of_interest["520"]["frames"] = 70

# Do we even have the XML for:
# ncslgr10a-003-fs-JOHN_FINISH_READ_2_BOOK_  (71 frames) [the XML file seems to have only one instance of this phrase]
# ncslgr10a-006-fs-JOHN_FUTURE_FINISH_READ_2_ (82 frames) [the XML file seems to have only one instance of this phrase]
# (the "ABOUT WHAT" appears 3 times in the XML, so we probably do... but which is it?)

path = #insert your path to 'ncslgr-xml/'
files = [ f for f in listdir(path) if isfile(join(path,f)) ]

for fn in ["ncslgr10a.xml"]: #eventually: files
    tree = ET.parse(path+fn) 
    root = tree.getroot()
    coding_scheme = populate_coding_scheme(root)
    
    for media_num in media_ref_of_interest.keys():
        for utterance in root.findall(".//UTTERANCE//*[@ID='" + media_num + "'].."):
            utter_rep = populate_utterance_representation(utterance, coding_scheme)
            write_outputs(path + "../../", fn, utter_rep)
