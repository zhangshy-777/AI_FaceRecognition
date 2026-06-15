import json
import os

via_json_path = "./via_project.json"
output_jsonl = "./annotations.jsonl"

def main():
    with open(via_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_meta = data.get("_via_img_metadata", {})
    lines = []

    for _, info in img_meta.items():
        fname = info.get("filename", "")
        regions = info.get("regions", [])
        faces = []

        for reg in regions:
            shape = reg.get("shape_attributes", {})
            x = shape.get("x", 0)
            y = shape.get("y", 0)
            w = shape.get("width", 0)
            h = shape.get("height", 0)

            # 过滤无效小框
            if w < 20 or h < 20:
                continue

            attr = reg.get("region_attributes", {})
            # 去除换行、空格，空值统一设为 unknown
            label = attr.get("identity_id", "").strip()
            if not label:
                label = "unknown"

            faces.append({
                "identity_id": label,
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        if not faces:
            continue

        img_type = "single" if len(faces) == 1 else "multi"
        item = {
            "image": f"images/{fname}",
            "image_type": img_type,
            "faces": faces
        }
        lines.append(json.dumps(item, ensure_ascii=False))

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"✅ 转换完成！共生成 {len(lines)} 条标注")
    print(f"输出文件：{output_jsonl}")

if __name__ == "__main__":
    main()