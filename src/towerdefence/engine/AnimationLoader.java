package towerdefence.engine;

import org.newdawn.slick.Animation;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class AnimationLoader {

    // Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BOSS = 3;
    // Direction constants
    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;
    
    private Animation up, down, left, right;

    public AnimationLoader() {
    }

    public Animation[] getCritterAnimation(int critterType) throws SlickException {
        Image normalSheet = null;
        if (critterType == NORMAL) {
            normalSheet = new Image("data/sprites/critters/antNormal.png");
        } else if (critterType == FIRE) {
            normalSheet = new Image("data/sprites/critters/antFire.png");
        } else if (critterType == ICE) {
            normalSheet = new Image("data/sprites/critters/antIce.png");
        }
        SpriteSheet critterSheet = new SpriteSheet(normalSheet, 64, 64);
        Image[] movementLeft = {critterSheet.getSprite(0, 0), critterSheet.getSprite(1, 0),
            critterSheet.getSprite(2, 0), critterSheet.getSprite(3, 0),
            critterSheet.getSprite(4, 0), critterSheet.getSprite(5, 0),
            critterSheet.getSprite(6, 0), critterSheet.getSprite(7, 0)};
        Image[] movementUp = {critterSheet.getSprite(0, 2), critterSheet.getSprite(1, 2),
            critterSheet.getSprite(2, 2), critterSheet.getSprite(3, 2),
            critterSheet.getSprite(4, 2), critterSheet.getSprite(5, 2),
            critterSheet.getSprite(6, 2), critterSheet.getSprite(7, 2)};
        Image[] movementRight = {critterSheet.getSprite(0, 4), critterSheet.getSprite(1, 4),
            critterSheet.getSprite(2, 4), critterSheet.getSprite(3, 4),
            critterSheet.getSprite(4, 4), critterSheet.getSprite(5, 4),
            critterSheet.getSprite(6, 4), critterSheet.getSprite(7, 4)};
        Image[] movementDown = {critterSheet.getSprite(0, 6), critterSheet.getSprite(1, 6),
            critterSheet.getSprite(2, 6), critterSheet.getSprite(3, 6),
            critterSheet.getSprite(4, 6), critterSheet.getSprite(5, 6),
            critterSheet.getSprite(6, 6), critterSheet.getSprite(7, 6)};
        left = new Animation(movementLeft, 100, true);
        up = new Animation(movementUp, 100, true);
        right = new Animation(movementRight, 100, true);
        down = new Animation(movementDown, 100, true);

        Animation[] critterAnimation = {up,down,left,right};

        return critterAnimation;
    }
}

