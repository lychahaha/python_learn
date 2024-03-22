import socket
import select

# tcp
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET是ipv4，AF_INET6是ipv6
server.bind(('localhost',8001))
server.listen(5)
conn,addr = server.accept() #conn和下面的client用法一样,addr是(ip,port)格式

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost',8001))
client.send(b'hehe')
data = client.recv(20) #有东西就返回，最多返回20个，而不是等20个才返回
client.close()


# udp
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('localhost',8001))
data,addr = server.recvfrom(20)
server.sendto(b'hehe', addr)

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.sendto(b'hehe', ('localhost',8001))
data,addr = client.recvfrom(20)

# 其他
conn.getpeername() #返回对方的addr
conn.getsockname() #返回自己的addr



# IO多路复用(epoll)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #设置IP地址复用？
server.bind(('localhost',8001))
server.listen(5)
server.setblocking(False) #设置非阻塞

epoll = select.epoll()
epoll.register(server.fileno(), select.EPOLLIN)
fd2sock = {server.fileno():server}
msg_q = {}
is_break = False

while True:
    events = epoll.poll(10) #10秒超时
    if not events:
        continue
    if is_break:
        break

    for fd,event in events:
        sock = fd2sock[fd]
        if sock == server:
            conn, addr = server.accept()
            conn.setblocking(False)
            epoll.register(conn.fileno(), select.EPOLLIN)
            fd2sock[conn.fileno()] = conn
            msg_q[conn] = []
        elif event & select.EPOLLHUP:
            epoll.unregister(fd)
            fd2sock[fd].close()
            fd2sock.pop(fd)
        elif event & select.EPOLLIN:
            data = socket.recv(1024)
            msg_q[sock].append(data)
            epoll.modify(fd, select.EPOLLOUT)
        elif event & select.EPOLLOUT:
            data = msg_q[sock].pop()
            socket.send(data)
            epoll.modify(fd, select.EPOLLIN)

epoll.unregister(server.fileno())
epoll.close()
server.close()