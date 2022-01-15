import qrcode
link = input("Enter a link to convert to QR Code: ")
img = qrcode.make(link)
img.save("qrcode.jpg")
