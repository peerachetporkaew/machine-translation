from pycnnum import *
import pythainlp.util as util 

def MONEYTh(text):
    print("MONEYTh")

    text = text.split("_")
    num = text[0].split("@")[0]
    cur = text[1].split("@")[0]

    if cur == "บาท":
        return num + "泰铢"
    
    if cur == "ดอลลาร์":
        return num + "美元"
    
    if cur == "เยน":
        return num + "日元"

    if cur == "หยวน":
        return num + "元"
    return text

def MONEYZh(text):
    
    if text.endswith("美元"):
        currency = "ดอลลาร์"
        text = text[:-2]
    elif text.endswith("泰铢"):
        currency = "บาท"
        text = text[:-2]
    elif text.endswith("日元"):
        currency = "เยน"
        text = text[:-2]
    elif text.endswith("元"):
        currency = "หยวน"
        text = text[:-1]
    if text.isnumeric():
        digit = int(text)
    else:
        digit = cn2num(text)
    print("DIGIT",digit)
    thdigit = util.num_to_thaiword(digit)

    if thdigit != "หนึ่ง" and thdigit.endswith("หนึ่ง"):
        thdigit = thdigit[:-5] + "เอ็ด"

    return thdigit + currency

def NUMZh(text):
    if text.isdigit():
        digit = float(text)
        if digit.is_integer():
            return util.num_to_thaiword(int(text))
        else:
            return text
    digit = cn2num(text)
    print("DIGIT",digit)
    return util.num_to_thaiword(digit)


def TIMEZh(text):
    if text.endswith("分"):
        text = text[:-1]
    text = text.split("点")
    if len(text[1]) > 0:
        if text[1] == "半":
            digit = str(cn2num(text[0])) + ":30น."
        else:
            digit = str(cn2num(text[0])) + ":" + str(cn2num(text[1])) + "น."
    else:
        digit = str(cn2num(text[0])) + ":00น."
    return digit

def todigit(text):
    text = text.replace("零","0")
    text = text.replace("一","1")
    text = text.replace("二","2")
    text = text.replace("三","3")
    text = text.replace("四","4")
    text = text.replace("五","5")
    text = text.replace("六","6")
    text = text.replace("七","7")
    text = text.replace("八","8")
    text = text.replace("九","9")

    return text

def DATEZh(text):

    monthStr = "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม"
    monthStr = monthStr.split(" ")
    if text == "明天":
        return "พรุ่งนี้"
    if text == "后天":
        return "วันมะรืนนี้"
    if text == "今天":
        return "วันนี้"

    year = ""
    if "年" in text:
        year = text.split("年")
        year, text = year[0], year[1]

        if year.isnumeric():
            year="ค.ศ." + todigit(year) 
        else:
            if year == "今":
                year = "ปีนี้"
            elif year == "明":
                year = "ปีหน้า"
            elif year == "去":
                year = "ปีที่แล้ว"
            else:
                year = " ค.ศ." + str(cn2num(year))
        
    month = ""
    print("M",text)
    if "月" in text: 
        month = text.split("月")
        month, text = month[0], month[1]
        if month.isnumeric():
            month = " เดือน" + monthStr[(int(month)-1)%12] + " "
        else:
            month = " เดือน" + monthStr[(cn2num(month)-1)%12] + " "
        
    day = ""
    if len(text) > 0:
        print(text)
        text = text.replace("日","").replace("号","")
        if text.isnumeric():
            day = "วันที่ " + text
        else:
            day = "วันที่ " + str(cn2num(text))
    
    return day+month+year
