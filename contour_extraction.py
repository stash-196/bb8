import cv2


# 物体認識関数
def extraction(frame1, range_bb8, range_obs, range_goal):  # frame1: カメラから読み取られた画像をresizeした画像
    # range_: それぞれの物体の面積の閾値
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # frame1をグレイスケール化

    # 二値化
    ret, th1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # ret:よくわからないが必要ない　th1: 二値化した画像

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours: 輪郭として抽出された点の座標の行列
    # hierarchy: よくわからないが必要ない
    # bb8の面積のもののみ選別
    bb8 = []  # bb8として認識された輪郭の座標を格納
    # bb8_areas = []  # bb8として認識された輪郭から計算された面積を格納
    obs = []  # 障害物として認識された輪郭の座標を格納
    # obs_areas = []  # 障害物として認識された輪郭から計算された面積を格納
    goal = []  # 目的地として認識された輪郭の座標を格納
    # goal_areas = []  # 目的地として認識された輪郭から計算された面積を格納
    prev_frame = frame1  # bb8->obstacle->goalの順での前の物体の輪郭画像

    for cnt in contours:
        area = cv2.contourArea(cnt)  # 全輪郭の面積計算
        # bb8の面積のもののみ選別
        if range_bb8[0] < area < range_bb8[1]:
            bb8.append(cnt)  # bb8に輪郭座標を格納
            # bb8_areas.append(area)  # 面積を格納

        # 障害物の面積のもののみ選別
        if range_obs[0] < area < range_obs[1]:
            epsilon = 0.1 * cv2.arcLength(cnt, True)  # 輪郭近似
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            obs.append(approx)
            # obs_areas.append(area)

        # ゴールの面積のもののみ選別
        if range_goal[0] < area < range_goal[1]:
            epsilon = 0.1 * cv2.arcLength(cnt, True)  # 輪郭近似
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            goal.append(approx)
            # goal_areas.append(area)

    # BB8　認識判定
    if len(bb8) == 0:
        print("BB8を認識できません")  # prev_frame = frame1
    # BB8の輪郭とその外接円の描画
    if len(bb8) == 1:
        bb8_frame = cv2.drawContours(frame1, bb8, -1, (0, 0, 255), 3)  # bb8の輪郭を赤色でframe1に追加

        for cnt in bb8:
            (x, y), radius = cv2.minEnclosingCircle(cnt) # bb8の輪郭に外接する円の中心座標と半径
            center = (int(x), int(y))
            radius = int(radius)
        #  print('<<BB8>> 面積：{} 座標：{}'.format(bb8_areas, center))

        prev_frame = cv2.circle(bb8_frame, center, radius, (0, 255, 0), 2) # 外接円をbb8の輪郭画像に追加しprev_frameを更新
        # 描画用
        # cv2.imshow('BB8 Frame',prev_frame)

    # 障害物　認識判定
    if len(obs) == 0:
        print("障害物を認識できません")
    if len(obs) != 0:
        # print('<<障害物>> 数：{}, 面積：{}, 座標：{}'.format(len(obs_areas), obs_areas, obs))
        # 障害物の輪郭の描画
        obs_frame = cv2.drawContours(prev_frame, obs, -1, (255, 0, 0), 3)  # prev_frame = bb8circle_frame + bb8_frame

        # 障害物の頂点の描画
        for cnt in obs:
            for p in cnt:
                prev_frame = cv2.circle(obs_frame, (p[0][0], p[0][1]), 8, (255, 255, 0), 4) # obstacleの４頂点をobs_frameに追加しprev_frameを更新
        # 描画用
        # cv2.imshow('Obstacle Frame',prev_frame)

        # ゴール 認識判定
    if len(goal) == 0:
        print("目的地を認識できません")
        cv2.imshow('Extraction Frame', prev_frame)
    if len(goal) != 0:
        # ゴールの輪郭の追加
        # "print('<<目的地>>　面積：{}, 座標：{}'.format(goal_areas, goal))
        goal_frame = cv2.drawContours(prev_frame, goal, -1, (128, 0, 128),
                                      3)  # prev_frame = bb8circle_frame + bb8_frame + obs_frame + obscircle_frame

        # ゴールの頂点を追加
        for cnt in goal:
            for p in cnt:
                gpoint_frame = cv2.circle(goal_frame, (p[0][0], p[0][1]), 8, (0, 255, 255), 4)
            #  全体の描画
        cv2.imshow('Extraction Frame', gpoint_frame)
    return bb8, obs, goal  # bb8, obstacle, goal の座標


if __name__ == '__extraction__':
    extraction()

# VideoCapture オブジェクトを取得
capture = cv2.VideoCapture(1)  # 引数 0:インカメラ 1:USBに接続しているカメラ

while True:
    ret, frame = capture.read()  # ret:画像が読み込めたか(true / false) frame:ウェブカメラから読み取った画像
    frame1 = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))  # frameを1/2のサイズに縮小
    frame1 = frame1[0:540, 280:960]  # 壁が入らないよいうにframe画像の左側280pxを切り取る
    range_bb8 = [300, 350]  # range_: それぞれの物体の面積の閾値
    range_obs = [2800, 3400]
    range_goal = [4000, 4300]
    # 関数呼び出し
    extraction(frame1, range_bb8, range_obs, range_goal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# キャプチャをリリースして、ウインドウを閉じる
capture.release()
cv2.destroyAllWindows()
