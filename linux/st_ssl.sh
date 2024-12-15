# openssl

## 生成密钥
openssl genrsa -out pri.key 2048 #生成私钥
    -aes256 #加密私钥
    -passout pass:abc #加密的密码
openssl rsa -in pri.key -pubout -out pub.key #生成公钥
    -passin pass:abc #加密私钥的密码

## 生成自签名证书
openssl req -new -x509 -days 365 -key pri.key -out cert.crt #使用已有私钥
openssl req -newkey rsa:2048 -nodes -keyout pri.key -x509 -days365 -out cert.crt
    -x509 #生成证书
    -days #证书有效期

    -newkey rsa:2048 #设置新私钥
    -nodes #私钥不设置
    
## 生成签名请求
openssl req -new -key pri.key -out server.csr

## CA认证
openssl x509 -req -days 365 -in server.csr -CA CAcert.crt -CAkey CApri.key -out cert.crt # PS:本地目录需要有一个cert.srl文件，并且里面写入"01"

## CA认证(生成v3证书)
openssl x509 -req -days 365 -in server.csr -CA CAcert.crt -CAkey CApri.key -out cert.crt -extfile openssl.cnf -extensions v3_req
### openssl.cnf从openssl安装目录拷贝出来(使用openssl version -d可以找到安装目录),并且在里面的[ v3_req ]的下方添加"subjectAltName = @alt_names",然后在结尾添加下列内容:
### [ alt_names ]
### DNS.1 = www.a.com
### IP.1 = 12.34.56.78
### IP.2 = 12.34.56.79


## 格式转换
openssl rsa -in pri.key -passin pass:abc -out pri2.key #私钥加密转非加密
openssl rsa -in pri.key -aes 256 -passout pass:abc -out pri2.key #私钥非加密转加密
openssl rsa -in pri.key -outform der-out pri.der #私钥PEM格式转DER格式
openssl x509 -in cert.cer -inform der -outform pem -out cert.pem #证书DER转PEM格式
openssl pkcs12 -export -inkey client.key -in client.crt -out client.pfx #证书CRT转PFX格式(微软客户端用)

## 查看信息
openssl rsa -in pri.key -noout -text #查看私钥具体信息
openssl req -in server.csr -noout -text #查看签名请求具体信息
openssl x509 -in cert.crt -noout -text #查看证书具体信息
